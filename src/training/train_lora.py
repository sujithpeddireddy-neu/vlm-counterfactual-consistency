# train_lora.py - LoRA fine-tuning of LLaVA-1.5 with pairwise consistency loss
import argparse
import json
import logging
import os
import sys
from pathlib import Path

import torch

# Reduce CUDA memory fragmentation — helps 4-bit QLoRA on small VRAM
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
from PIL import Image
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import DataLoader, Dataset
from transformers import AutoProcessor, BitsAndBytesConfig, LlavaForConditionalGeneration

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

MODEL_ID = "llava-hf/llava-1.5-7b-hf"
IGNORE_INDEX = -100
ASSISTANT_MARKER = "ASSISTANT:"


# Dataset
class CounterfactualPairDataset(Dataset):
    # dataset that flattens counterfactual families into (original, counterfactual) pairs

    def __init__(self, families_path, images_dir):
        with open(families_path, encoding="utf-8") as f:
            families = json.load(f)

        self.images_dir = Path(images_dir)
        self.pairs: List[Dict] = []

        for family in families:
            original = family["original"]
            image_id = str(family["image_id"])
            for cf in family["counterfactuals"]:
                self.pairs.append(
                    {
                        "image_id": image_id,
                        "orig_question": original["question"],
                        "orig_answer": str(original["answer"]).strip().lower(),
                        "cf_question": cf["counterfactual_question"],
                        "cf_expected_answer": str(cf.get("expected_answer", "")).strip().lower(),
                        "logical_relation": cf.get("logical_relation", ""),
                    }
                )

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        item = dict(self.pairs[idx])
        img_path = self._find_image(item["image_id"])
        item["image"] = Image.open(img_path).convert("RGB") if img_path else None
        return item

    def _find_image(self, image_id):
        for ext in (".jpg", ".jpeg", ".png"):
            path = self.images_dir / f"{image_id}{ext}"
            if path.exists():
                return str(path)
        return None


def encode_qa(processor, image, question, answer, device):
    # tokenize (image, question, answer) for teacher-forced training, returns enc and answer_start
    prompt_text = f"USER: <image>\n{question}\n{ASSISTANT_MARKER}"  # no trailing space
    full_text = prompt_text + " " + answer                           # space goes with answer

    processor_kwargs = dict(return_tensors="pt")
    if image is not None:
        enc = processor(text=full_text, images=image, **processor_kwargs)
        # Encode prompt alone WITH image so image tokens are counted correctly
        enc_prompt = processor(text=prompt_text, images=image, **processor_kwargs)
    else:
        enc = processor(text=full_text, **processor_kwargs)
        enc_prompt = processor(text=prompt_text, **processor_kwargs)

    # answer_start is the index of the first answer token.
    # prompt_len counts all prompt tokens (including image tokens).
    # If, despite the fix, full_text tokenizes to the same length as prompt_text
    # (e.g. answer is empty), fall back to prompt_len - 1 so at least 1 token
    # is included in the CE computation.
    prompt_len = enc_prompt["input_ids"].shape[1]
    full_len   = enc["input_ids"].shape[1]
    answer_start = prompt_len if full_len > prompt_len else max(0, prompt_len - 1)

    enc = {k: v.to(device) for k, v in enc.items()}

    # Mask the prompt in labels so CE is only computed on the answer tokens
    labels = enc["input_ids"].clone()
    labels[:, :answer_start] = IGNORE_INDEX
    enc["labels"] = labels

    return enc, answer_start


# Pairwise consistency loss
def pairwise_consistency_loss(logits_orig, logits_cf, answer_start_orig, answer_start_cf, logical_relation, yes_id, no_id):
    # penalize logically inconsistent yes/no answer pairs based on softmax distributions
    # Clamp to avoid out-of-bounds on very short sequences
    pos_orig = max(0, min(answer_start_orig - 1, logits_orig.size(1) - 1))
    pos_cf   = max(0, min(answer_start_cf   - 1, logits_cf.size(1)   - 1))

    probs_orig = torch.softmax(logits_orig[0, pos_orig], dim=-1)
    probs_cf   = torch.softmax(logits_cf[0, pos_cf],     dim=-1)

    p_yes_orig, p_no_orig = probs_orig[yes_id], probs_orig[no_id]
    p_yes_cf,   p_no_cf   = probs_cf[yes_id],   probs_cf[no_id]

    if logical_relation == "contradiction":
        # penalize agreement: P(yes|O)·P(yes|CF) + P(no|O)·P(no|CF)
        # Loss -> 0 when one says yes and the other says no (correct)
        # Loss -> 1 when both say the same thing (incorrect)
        loss = p_yes_orig * p_yes_cf + p_no_orig * p_no_cf

    elif logical_relation == "entails":
        # The counterfactual is a weaker claim entailed by the original, so
        # a truthful model should always answer "yes" to the CF question.
        loss = -torch.log(p_yes_cf + 1e-8)

    else:
        # attribute_change / object_change / spatial_change:
        # answers should differ — loss = 0 when maximally different, 1 when identical
        loss = 1.0 - torch.abs(p_yes_orig - p_yes_cf)

    return loss


# LoRA model
_VRAM_THRESHOLD_GB = 12.0


def _use_4bit(device):
    if device.type != "cuda":
        return False
    return torch.cuda.get_device_properties(0).total_memory / 1e9 < _VRAM_THRESHOLD_GB


def build_lora_model(model_id, lora_r, lora_alpha, lora_dropout, device):
    log.info("Loading %s : ", model_id)
    if _use_4bit(device):
        log.info("VRAM < %.0f GB — loading in 4-bit QLoRA (bitsandbytes)", _VRAM_THRESHOLD_GB)
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model = LlavaForConditionalGeneration.from_pretrained(
            model_id, quantization_config=bnb_cfg, low_cpu_mem_usage=True, device_map="auto"
        )
        # Required for QLoRA: enables gradient flow through quantized layers
        # gradient_checkpointing=True trades compute for memory (crucial on 8 GB VRAM)
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        # peft needs this hook when gradient checkpointing is active
        model.enable_input_require_grads()
    else:
        dtype = torch.float16 if device.type == "cuda" else torch.float32
        model = LlavaForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=dtype, low_cpu_mem_usage=True
        )
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["q_proj", "v_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    # When device_map="auto" is used (4-bit path), the model is already placed;
    # calling .to(device) again would fail for quantized models.
    if not _use_4bit(device):
        model = model.to(device)
    return model


# Training loop
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Using device: %s", device)

    model = build_lora_model(
        MODEL_ID,
        lora_r=args.lora_r,
        lora_alpha=args.lora_r * 2,
        lora_dropout=0.05,
        device=device,
    )
    # With device_map="auto" (4-bit path) the model spans devices; use first-param device for tensors.
    tensor_device = next(model.parameters()).device
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    # Token IDs for "yes" and "no" used in pairwise consistency loss
    yes_id = int(processor.tokenizer.convert_tokens_to_ids("yes"))
    no_id  = int(processor.tokenizer.convert_tokens_to_ids("no"))

    dataset = CounterfactualPairDataset(args.families, args.images)
    log.info("Dataset: %d (original, counterfactual) pairs", len(dataset))

    # batch_size=1: each step processes one pair; collate_fn returns the raw dict
    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0,
                        collate_fn=lambda batch: batch[0])

    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=args.lr,
        weight_decay=0.01,
    )

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    epoch_logs = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_ce_loss = total_pc_loss = 0.0
        num_steps = 0

        for step, item in enumerate(loader):
            if item["image"] is None:
                continue  # image missing; skip silently

            optimizer.zero_grad()

            # original question
            enc_orig, answer_start_orig = encode_qa(
                processor, item["image"],
                item["orig_question"], item["orig_answer"],
                tensor_device,
            )
            out_orig = model(**enc_orig)
            ce_orig = out_orig.loss  # CE on original answer tokens

            # counterfactual question — also train on it directly as a QA pair
            # so the model learns to answer CFs correctly, not just consistently
            cf_answer = item["cf_expected_answer"]
            has_cf_target = bool(cf_answer) and cf_answer not in {"unknown", "n/a", ""}

            enc_cf, answer_start_cf = encode_qa(
                processor, item["image"],
                item["cf_question"],
                cf_answer if has_cf_target else "yes",  # placeholder when target unknown
                tensor_device,
            )
            out_cf = model(**enc_cf)
            # ce_aug: cross-entropy on the CF as an independent QA training example
            ce_aug = out_cf.loss if has_cf_target else torch.tensor(0.0, device=tensor_device)

            # pairwise consistency loss
            pc_loss = pairwise_consistency_loss(
                out_orig.logits, out_cf.logits,
                answer_start_orig, answer_start_cf,
                item["logical_relation"],
                yes_id, no_id,
            )

            # Guard: skip if CE is NaN (all answer tokens were masked — empty answer)
            if torch.isnan(ce_orig):
                log.warning("NaN CE at step %d (empty answer?); skipping backward.", step)
                continue

            # Total loss = standard CE + data augmentation CE + pairwise consistency loss
            total_loss = (ce_orig + args.augmentation_weight * ce_aug) + args.consistency_weight * pc_loss
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_ce_loss += (ce_orig + args.augmentation_weight * ce_aug).item()
            total_pc_loss += pc_loss.item()
            num_steps += 1

            if (step + 1) % args.log_every == 0:
                log.info(
                    "Epoch %d  step %d/%d  CE=%.4f  PC=%.4f",
                    epoch, step + 1, len(dataset),
                    total_ce_loss / num_steps, total_pc_loss / num_steps,
                )

        avg_ce = total_ce_loss / max(num_steps, 1)
        avg_pc = total_pc_loss / max(num_steps, 1)
        log.info(
            "Epoch %d done — avg CE=%.4f  avg PC=%.4f  steps=%d",
            epoch, avg_ce, avg_pc, num_steps,
        )
        epoch_logs.append(
            {
                "epoch": epoch,
                "avg_ce_loss": round(avg_ce, 6),
                "avg_pairwise_consistency_loss": round(avg_pc, 6),
                "steps": num_steps,
            }
        )

        # Save LoRA adapter checkpoint
        ckpt_dir = output_dir / f"epoch_{epoch}"
        model.save_pretrained(str(ckpt_dir))
        processor.save_pretrained(str(ckpt_dir))
        log.info("Checkpoint saved -> %s", ckpt_dir)

    # Persist training metrics for the final report
    log_path = output_dir / "training_log.json"
    with log_path.open("w", encoding="utf-8") as f:
        json.dump(epoch_logs, f, indent=2)
    log.info("Training log saved -> %s", log_path)


# Entry point
def main():
    parser = argparse.ArgumentParser(
        description="LoRA fine-tune LLaVA-1.5 with pairwise consistency loss."
    )
    parser.add_argument("--families",           required=True,
                        help="Counterfactual families JSON")
    parser.add_argument("--images",             required=True,
                        help="Directory containing dataset images")
    parser.add_argument("--output",             default="results/checkpoints/lora_llava",
                        help="Directory for LoRA checkpoints and training log")
    parser.add_argument("--epochs",             type=int,   default=3)
    parser.add_argument("--lr",                 type=float, default=2e-4)
    parser.add_argument("--lora-r",             type=int,   default=8, dest="lora_r",
                        help="LoRA rank (default 8 for 8 GB VRAM; use 16 on larger GPUs)")
    parser.add_argument("--augmentation-weight", type=float, default=1.0, dest="augmentation_weight",
                        help="Weight for CF data augmentation CE loss (0 = disable augmentation)")
    parser.add_argument("--consistency-weight", type=float, default=0.5, dest="consistency_weight",
                        help="Lambda for pairwise consistency loss (0 = CE only)")
    parser.add_argument("--log-every",          type=int,   default=50, dest="log_every",
                        help="Log training metrics every N steps")
    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
