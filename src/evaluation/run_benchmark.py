# run_benchmark.py - end-to-end VLM benchmarking on counterfactual families
import argparse
import copy
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

# Allow running from the repo root without installing as a package
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.evaluation.consistency_score import score_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# Model loading
def load_model_runner(model_name, lora_checkpoint=None):
    if model_name == "llava":
        from src.models.run_llava import LlavaRunner
        log.info("Loading LLaVA-1.5%s ", f" + LoRA from {lora_checkpoint}" if lora_checkpoint else "")
        return LlavaRunner(lora_checkpoint=lora_checkpoint)
    if model_name == "instructblip":
        from src.models.run_instructblip import InstructBlipRunner
        log.info("Loading InstructBLIP : ")
        return InstructBlipRunner()
    raise ValueError(f"Unknown model {model_name!r}. Choose 'llava' or 'instructblip'.")


# Inference
def resolve_image_path(images_dir, image_id):
    base = Path(images_dir)
    for ext in (".jpg", ".jpeg", ".png"):
        p = base / f"{image_id}{ext}"
        if p.exists():
            return str(p)
    return None


def run_inference(families, runner, images_dir):
    # deep-copy families and run model inference on each question
    families = copy.deepcopy(families)
    total = len(families)

    for i, family in enumerate(families):
        img = resolve_image_path(images_dir, str(family["image_id"]))
        if img is None:
            log.warning("Image not found for image_id=%s — predictions set to N/A", family["image_id"])

        def predict(question):
            return runner.answer_question(img, question) if img else "N/A"

        family["original"]["model_prediction"] = predict(family["original"]["question"])
        for cf in family["counterfactuals"]:
            cf["model_prediction"] = predict(cf["counterfactual_question"])

        if (i + 1) % 50 == 0 or (i + 1) == total:
            log.info("  Inference: %d / %d families done", i + 1, total)

    return families


# Analysis
def attach_question_types(scored, families):
    # patch question_type into scored results so compute_breakdown can group by it
    qtype_map = {
        family["question_id"]: family["original"].get("question_type", "unknown")
        for family in families
    }
    for result in scored["family_results"]:
        result["question_type"] = qtype_map.get(result["question_id"], "unknown")
    return scored


def compute_breakdown(scored):
    # compute per-question-type and per-intervention-type pass rates
    by_qtype_consistency = defaultdict(list)
    by_qtype_correct = defaultdict(list)
    by_intervention = defaultdict(list)

    for result in scored["family_results"]:
        qtype = result.get("question_type", "unknown")
        by_qtype_consistency[qtype].append(result["family_consistency_score"])
        by_qtype_correct[qtype].append(int(result.get("original_correct", False)))
        for detail in result["details"]:
            by_intervention[detail["intervention_type"]].append(int(detail["passed"]))

    return {
        "by_question_type": {
            k: {
                "mean_consistency_score": round(sum(by_qtype_consistency[k]) / len(by_qtype_consistency[k]), 4),
                "vqa_accuracy": round(sum(by_qtype_correct[k]) / len(by_qtype_correct[k]), 4),
                "num_families": len(by_qtype_consistency[k]),
            }
            for k in sorted(by_qtype_consistency)
        },
        "by_intervention_type": {
            itype: {
                "pass_rate": round(sum(pass_results) / len(pass_results), 4),
                "passed": sum(pass_results),
                "num_counterfactuals": len(pass_results),
            }
            for itype, pass_results in sorted(by_intervention.items())
        },
    }


# Output
def save_json(obj, path):
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    log.info("Saved -> %s", output_path)


def print_report(model_name, scored, breakdown):
    print(f"\nBenchmark results — {model_name}")
    print(f"  Families evaluated       : {scored['num_families']}")
    print(f"  Dataset Consistency Score: {scored['dataset_consistency_score']:.4f}")
    print(f"  Standard VQA Accuracy    : {scored.get('vqa_accuracy', 'N/A')}")

    print("\n  By question type:")
    for qtype, stats in breakdown["by_question_type"].items():
        print(
            f"    {qtype:<22} consistency={stats['mean_consistency_score']:.4f}"
            f"  vqa_acc={stats['vqa_accuracy']:.4f}"
            f"  n={stats['num_families']}"
        )

    print("\n  By intervention type:")
    for itype, stats in breakdown["by_intervention_type"].items():
        print(
            f"    {itype:<26} pass_rate={stats['pass_rate']:.4f}"
            f"  passed={stats['passed']}/{stats['num_counterfactuals']}"
        )
    print()


# Entry point
def main():
    parser = argparse.ArgumentParser(
        description="Benchmark a VLM on counterfactual families and report consistency."
    )
    parser.add_argument("--families",   required=True, help="Counterfactual families JSON")
    parser.add_argument("--images",     required=True, help="Directory containing dataset images")
    parser.add_argument("--model",      required=True, choices=["llava", "instructblip"])
    parser.add_argument("--output",     default="results/", help="Root directory for output files")
    parser.add_argument("--checkpoint", default=None,
                        help="Optional LoRA adapter checkpoint dir to load on top of the base model")
    parser.add_argument("--output-prefix", default=None, dest="output_prefix",
                        help="Prefix for output filenames (default: model name, or model_lora if --checkpoint given)")
    args = parser.parse_args()

    prefix = args.output_prefix or (f"{args.model}_lora" if args.checkpoint else args.model)

    # Load families
    with open(args.families, encoding="utf-8") as f:
        families = json.load(f)
    log.info("Loaded %d families from %s", len(families), args.families)

    # Inference
    runner = load_model_runner(args.model, lora_checkpoint=args.checkpoint)
    families_with_preds = run_inference(families, runner, args.images)

    # Save raw predictions
    preds_path = str(Path(args.output) / "predictions" / f"{prefix}_predictions.json")
    save_json(families_with_preds, preds_path)

    # Score
    scored = score_dataset(families_with_preds)
    attach_question_types(scored, families_with_preds)

    # Breakdown
    breakdown = compute_breakdown(scored)
    scored["breakdown"] = breakdown

    # Save scored results
    scored_path = str(Path(args.output) / "metrics" / f"{prefix}_consistency_scores.json")
    save_json(scored, scored_path)

    # Print human-readable summary
    print_report(prefix, scored, breakdown)


if __name__ == "__main__":
    main()
