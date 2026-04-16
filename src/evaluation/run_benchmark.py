"""
run_benchmark.py — end-to-end VLM benchmarking on counterfactual families.

Runs model inference on every question in every family, scores consistency,
and breaks results down by question_type and intervention_type.

Usage:
    python src/evaluation/run_benchmark.py \
        --families data/counterfactual/gqa_sample_counterfactuals.json \
        --images   data/raw/gqa_sample/images \
        --model    llava \
        --output   results/
"""
import argparse
import copy
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

# Allow running from the repo root without installing as a package
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.evaluation.consistency_score import score_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model_runner(model_name: str, lora_checkpoint: Optional[str] = None):
    if model_name == "llava":
        from src.models.run_llava import LlavaRunner
        log.info("Loading LLaVA-1.5%s ...", f" + LoRA from {lora_checkpoint}" if lora_checkpoint else "")
        return LlavaRunner(lora_checkpoint=lora_checkpoint)
    if model_name == "instructblip":
        from src.models.run_instructblip import InstructBlipRunner
        log.info("Loading InstructBLIP ...")
        return InstructBlipRunner()
    raise ValueError(f"Unknown model {model_name!r}. Choose 'llava' or 'instructblip'.")


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def resolve_image_path(images_dir: str, image_id: str) -> Optional[str]:
    base = Path(images_dir)
    for ext in (".jpg", ".jpeg", ".png"):
        p = base / f"{image_id}{ext}"
        if p.exists():
            return str(p)
    return None


def run_inference(families: List[Dict], runner, images_dir: str) -> List[Dict]:
    """
    Deep-copy families and annotate every question with a model_prediction field.
    Missing images are logged as warnings; their predictions are set to "N/A".
    """
    families = copy.deepcopy(families)
    total = len(families)

    for i, family in enumerate(families):
        img = resolve_image_path(images_dir, str(family["image_id"]))
        if img is None:
            log.warning("Image not found for image_id=%s — predictions set to N/A", family["image_id"])

        def predict(question: str) -> str:
            return runner.answer_question(img, question) if img else "N/A"

        family["original"]["model_prediction"] = predict(family["original"]["question"])
        for cf in family["counterfactuals"]:
            cf["model_prediction"] = predict(cf["counterfactual_question"])

        if (i + 1) % 50 == 0 or (i + 1) == total:
            log.info("  Inference: %d / %d families done", i + 1, total)

    return families


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def attach_question_types(scored: Dict, families: List[Dict]) -> Dict:
    """
    score_dataset() doesn't propagate question_type. Patch it in from the
    original families so compute_breakdown() can group by it.
    """
    qtype_map = {
        f["question_id"]: f["original"].get("question_type", "unknown")
        for f in families
    }
    for result in scored["family_results"]:
        result["question_type"] = qtype_map.get(result["question_id"], "unknown")
    return scored


def compute_breakdown(scored: Dict) -> Dict:
    """
    Returns per-question-type and per-intervention-type statistics,
    including both Consistency Score and standard VQA accuracy per type.
    """
    by_qtype_consistency: Dict[str, List[float]] = defaultdict(list)
    by_qtype_correct: Dict[str, List[int]] = defaultdict(list)
    by_intervention: Dict[str, List[int]] = defaultdict(list)

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
            k: {
                "pass_rate": round(sum(v) / len(v), 4),
                "passed": sum(v),
                "num_counterfactuals": len(v),
            }
            for k, v in sorted(by_intervention.items())
        },
    }


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def save_json(obj, path: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    log.info("Saved -> %s", p)


def print_report(model_name: str, scored: Dict, breakdown: Dict) -> None:
    sep = "=" * 64
    print(f"\n{sep}")
    print(f"  Benchmark Report  |  model: {model_name}")
    print(sep)
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
    print(sep + "\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
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
