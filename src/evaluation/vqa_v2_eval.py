# vqa_v2_eval.py - evaluate LLaVA-1.5 on VQA v2 validation split
import argparse
import json
import logging
import os
import sys
from pathlib import Path

import torch
from PIL import Image

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from transformers import AutoProcessor, BitsAndBytesConfig, LlavaForConditionalGeneration

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.evaluation.consistency_score import extract_yesno, normalize_text

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

MODEL_ID = "llava-hf/llava-1.5-7b-hf"
_VRAM_THRESHOLD_GB = 12.0


def _use_4bit():
    if not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_properties(0).total_memory / 1e9 < _VRAM_THRESHOLD_GB


def load_model(lora_checkpoint=None):
    # load LLaVA-1.5, optionally with a LoRA adapter merged in
    log.info("Loading base model %s : ", MODEL_ID)
    if _use_4bit():
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model = LlavaForConditionalGeneration.from_pretrained(
            MODEL_ID, quantization_config=bnb_cfg, low_cpu_mem_usage=True, device_map="auto"
        )
    else:
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        model = LlavaForConditionalGeneration.from_pretrained(
            MODEL_ID, torch_dtype=dtype, low_cpu_mem_usage=True,
            device_map="auto" if torch.cuda.is_available() else None,
        )

    if lora_checkpoint:
        from peft import PeftModel
        log.info("Loading LoRA adapter from %s : ", lora_checkpoint)
        model = PeftModel.from_pretrained(model, lora_checkpoint)
        model = model.merge_and_unload()
        log.info("LoRA adapter merged into base model.")

    model.eval()
    return model


@torch.no_grad()
def answer_question(model, processor, image, question):
    prompt = f"USER: <image>\n{question}\nASSISTANT:"
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    target_device = next(model.parameters()).device
    inputs = {k: v.to(target_device) for k, v in inputs.items()}
    output_ids = model.generate(**inputs, max_new_tokens=50)
    full_text = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
    if "ASSISTANT:" in full_text:
        return full_text.split("ASSISTANT:")[-1].strip()
    return full_text.strip()


def vqa_accuracy(prediction, ground_truths):
    # official VQA v2 soft accuracy: min(matching_annotators / 3, 1.0)
    pred = normalize_text(prediction)
    # For yes/no predictions, extract bare yes/no from verbose outputs
    if pred.startswith("yes") or pred.startswith("no"):
        pred = extract_yesno(pred)
    count = sum(1 for gt in ground_truths if normalize_text(gt) == pred)
    return min(count / 3.0, 1.0)


def load_vqa_v2_streaming(split="validation", num_samples=500):
    # stream VQA v2 from HuggingFace, tries multiple dataset IDs in order
    try:
        from datasets import load_dataset
    except ImportError:
        raise RuntimeError("datasets package required: pip install datasets")

    # Candidate dataset IDs (in preference order).
    # 'lmms-lab/VQAv2' is parquet-based (no loading script) and includes images.
    candidates = [
        ("lmms-lab/VQAv2", {"split": split}),
        ("Multimodal-Fatima/VQAv2_val", {"split": "validation"}),
        ("merve/vqav2-small", {"split": "validation"}),
    ]

    dataset = None
    for ds_id, kwargs in candidates:
        try:
            log.info("Trying VQA v2 dataset: %s : ", ds_id)
            dataset = load_dataset(ds_id, streaming=True, **kwargs)
            log.info("Loaded: %s", ds_id)
            break
        except Exception as exc:
            log.warning("  %s failed: %s", ds_id, exc)

    if dataset is None:
        raise RuntimeError(
            "Could not load any VQA v2 dataset from HuggingFace. "
            "Try: huggingface-cli login  or  pip install -U datasets"
        )

    log.info("Streaming VQA v2 %s split (first %d samples) ...", split, num_samples)
    samples = []
    for i, item in enumerate(dataset):
        if i >= num_samples:
            break
        # Extract all annotator answers for soft accuracy
        answers = [a["answer"] for a in item.get("answers", [])]
        if not answers:
            answers = [item.get("multiple_choice_answer", "")]
        samples.append({
            "question_id": item.get("question_id", i),
            "question": item["question"],
            "image": item["image"],  # PIL.Image from HF dataset
            "ground_truth_answers": answers,
            "multiple_choice_answer": item.get("multiple_choice_answer", ""),
        })
        if (i + 1) % 100 == 0:
            log.info("  Loaded %d/%d samples ...", i + 1, num_samples)

    log.info("Loaded %d VQA v2 samples.", len(samples))
    return samples


def run_evaluation(num_samples, output_path, lora_checkpoint=None):
    model = load_model(lora_checkpoint)
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    samples = load_vqa_v2_streaming(num_samples=num_samples)

    results = []
    total_acc = 0.0
    yes_no_count = yes_no_acc = 0
    other_count = other_acc = 0

    for idx, sample in enumerate(samples):
        image = sample["image"].convert("RGB")
        question = sample["question"]
        gt_answers = sample["ground_truth_answers"]

        prediction = answer_question(model, processor, image, question)
        acc = vqa_accuracy(prediction, gt_answers)
        total_acc += acc

        # Track yes/no vs open-ended separately
        mc_answer = normalize_text(sample["multiple_choice_answer"])
        is_yes_no = mc_answer in {"yes", "no"}
        if is_yes_no:
            yes_no_count += 1
            yes_no_acc += acc
        else:
            other_count += 1
            other_acc += acc

        results.append({
            "question_id": sample["question_id"],
            "question": question,
            "prediction": prediction,
            "multiple_choice_answer": sample["multiple_choice_answer"],
            "soft_accuracy": round(acc, 4),
        })

        if (idx + 1) % 50 == 0:
            running_acc = total_acc / (idx + 1)
            log.info("  Step %d/%d  running_acc=%.4f", idx + 1, len(samples), running_acc)

    total_samples = len(results)
    overall_acc = total_acc / total_samples if total_samples else 0.0
    summary = {
        "model": MODEL_ID,
        "lora_checkpoint": lora_checkpoint,
        "num_samples": total_samples,
        "overall_vqa_accuracy": round(overall_acc, 4),
        "yes_no_accuracy": round(yes_no_acc / yes_no_count, 4) if yes_no_count else None,
        "open_ended_accuracy": round(other_acc / other_count, 4) if other_count else None,
        "yes_no_count": yes_no_count,
        "open_ended_count": other_count,
        "per_sample_results": results,
    }

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    log.info("Results saved -> %s", output_path)

    log.info("VQA v2 results:")
    log.info("  Overall accuracy : %.4f  (%d samples)", overall_acc, total_samples)
    if yes_no_count:
        log.info("  Yes/No accuracy  : %.4f  (%d questions)", yes_no_acc / yes_no_count, yes_no_count)
    if other_count:
        log.info("  Open-ended acc   : %.4f  (%d questions)", other_acc / other_count, other_count)

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate LLaVA-1.5 on VQA v2 (generalization test)."
    )
    parser.add_argument(
        "--lora-checkpoint", default=None, dest="lora_checkpoint",
        help="Path to LoRA adapter checkpoint directory (e.g. results/checkpoints/lora_llava/epoch_3). "
             "If omitted, evaluates the base model only.",
    )
    parser.add_argument(
        "--output", default="results/metrics/vqa_v2_base.json",
        help="Output path for the evaluation JSON",
    )
    parser.add_argument(
        "--num-samples", type=int, default=500, dest="num_samples",
        help="Number of VQA v2 validation samples to evaluate (default 500)",
    )
    args = parser.parse_args()

    run_evaluation(
        num_samples=args.num_samples,
        output_path=args.output,
        lora_checkpoint=args.lora_checkpoint,
    )


if __name__ == "__main__":
    main()
