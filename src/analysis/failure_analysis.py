"""
failure_analysis.py — Extract and display failure cases for the Week 8 report.

Identifies families where the model was least consistent and shows what it predicted
vs. what was logically required. Also surfaces per-intervention-type patterns.

Outputs:
  Console  — formatted failure report
  JSON     — results/analysis/<model>_failure_cases.json  (structured, for programmatic use)
  Text     — results/analysis/<model>_failure_report.txt  (ready to paste into report)

Usage:
    python src/analysis/failure_analysis.py \
        --results results/metrics/llava_consistency_scores.json \
        --top-n   20 \
        --output  results/analysis/
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_scored(path: str) -> Dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def infer_model_name(path: str) -> str:
    return Path(path).stem.split("_consistency")[0]


# ---------------------------------------------------------------------------
# Family-level analysis
# ---------------------------------------------------------------------------

def worst_families(scored: Dict, top_n: int) -> List[Dict]:
    """Return the `top_n` families with the lowest consistency score."""
    families = scored.get("family_results", [])
    return sorted(families, key=lambda f: f["family_consistency_score"])[:top_n]


def failure_summary_by_intervention(scored: Dict) -> Dict[str, Dict]:
    """
    Per-intervention-type failure statistics and one example of a failing case.
    """
    stats: Dict[str, Dict] = defaultdict(
        lambda: {"total": 0, "failed": 0, "examples": []}
    )
    for family in scored.get("family_results", []):
        orig_q = family.get("original_question", "")
        for detail in family.get("details", []):
            itype = detail["intervention_type"]
            stats[itype]["total"] += 1
            if not detail["passed"]:
                stats[itype]["failed"] += 1
                if len(stats[itype]["examples"]) < 3:
                    stats[itype]["examples"].append(
                        {
                            "original_question":    orig_q,
                            "counterfactual_question": detail.get("counterfactual_question", ""),
                            "expected_answer":      detail.get("expected_answer", ""),
                            "model_prediction":     detail.get("model_prediction", ""),
                            "reason":               detail.get("reason", ""),
                        }
                    )

    # Compute failure rate
    for itype, s in stats.items():
        s["failure_rate"] = round(s["failed"] / s["total"], 4) if s["total"] else 0.0
    return dict(sorted(stats.items(), key=lambda x: -x[1]["failure_rate"]))


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

TICK = "\u2714"   # ✔
CROSS = "\u2718"  # ✘
BAR   = "\u2500"  # ─
TOP   = "\u250c"  # ┌
MID   = "\u251c"  # ├
BOT   = "\u2514"  # └
VERT  = "\u2502"  # │


def _truncate(s: str, n: int = 90) -> str:
    return s if len(s) <= n else s[:n - 3] + "..."


def format_family(rank: int, family: Dict) -> str:
    lines = []
    score   = family["family_consistency_score"]
    passed  = family["passed_count"]
    total   = family["total_counterfactuals"]
    qtype   = family.get("question_type", "unknown")

    lines.append(f"{TOP}{BAR * 78}")
    lines.append(
        f"{VERT} #{rank:<3} [{qtype}]  Score: {score:.2f}  "
        f"({passed}/{total} passed)  id={family.get('question_id','')}"
    )
    lines.append(f"{VERT} Original Q:  {_truncate(family.get('original_question', ''))}")
    lines.append(
        f"{VERT} Answer: {family.get('original_answer', '')}  "
        f"| Model said: {_truncate(str(family.get('original_model_prediction', '')), 40)}"
    )
    lines.append(f"{MID}{BAR * 78}")

    for j, detail in enumerate(family.get("details", []), 1):
        ok   = detail["passed"]
        mark = TICK if ok else CROSS
        lines.append(
            f"{VERT}  CF#{j}  [{detail['intervention_type']} | {detail['logical_relation']}]  {mark}"
        )
        lines.append(f"{VERT}    Q:         {_truncate(detail.get('counterfactual_question', ''))}")
        lines.append(f"{VERT}    Expected:  {detail.get('expected_answer', '')}")
        lines.append(f"{VERT}    Predicted: {_truncate(str(detail.get('model_prediction', '')), 60)}")
        if not ok:
            lines.append(f"{VERT}    Reason:    {_truncate(detail.get('reason', ''), 70)}")
        if j < len(family.get("details", [])):
            lines.append(f"{VERT}  {'·' * 60}")

    lines.append(f"{BOT}{BAR * 78}")
    return "\n".join(lines)


def format_intervention_summary(stats: Dict[str, Dict]) -> str:
    lines = ["", "=" * 80, "  FAILURE PATTERN ANALYSIS — BY INTERVENTION TYPE", "=" * 80]
    for itype, s in stats.items():
        pct = s["failure_rate"] * 100
        bar = "\u2588" * int(pct / 5)   # █ block bar, max 20 chars for 100%
        lines.append(
            f"\n  {itype:<26}  fail={s['failed']:>4}/{s['total']:<4}  "
            f"({pct:5.1f}%)  {bar}"
        )
        for ex in s["examples"]:
            lines.append(f"    Example failure:")
            lines.append(f"      Orig:      {_truncate(ex['original_question'], 70)}")
            lines.append(f"      CF:        {_truncate(ex['counterfactual_question'], 70)}")
            lines.append(f"      Expected:  {ex['expected_answer']}")
            lines.append(f"      Got:       {_truncate(str(ex['model_prediction']), 60)}")
    lines.append("\n" + "=" * 80)
    return "\n".join(lines)


def format_header(model_name: str, scored: Dict, top_n: int) -> str:
    return (
        f"\n{'=' * 80}\n"
        f"  FAILURE CASE ANALYSIS  |  model: {model_name.upper()}\n"
        f"{'=' * 80}\n"
        f"  Total families:      {scored['num_families']}\n"
        f"  Dataset score:       {scored['dataset_consistency_score']:.4f}\n"
        f"  Showing worst {top_n} families\n"
        f"{'=' * 80}\n"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract and report failure cases from scored consistency results."
    )
    parser.add_argument("--results", required=True,
                        help="*_consistency_scores.json from run_benchmark.py")
    parser.add_argument("--top-n",   type=int, default=20, dest="top_n",
                        help="Number of worst families to display")
    parser.add_argument("--output",  default="results/analysis/")
    args = parser.parse_args()

    scored     = load_scored(args.results)
    model_name = infer_model_name(args.results)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Build report sections ──────────────────────────────────────────
    header = format_header(model_name, scored, args.top_n)

    worst   = worst_families(scored, args.top_n)
    family_blocks = "\n".join(
        format_family(rank + 1, fam) for rank, fam in enumerate(worst)
    )

    intervention_stats = failure_summary_by_intervention(scored)
    pattern_section    = format_intervention_summary(intervention_stats)

    full_report = header + "\n" + family_blocks + "\n" + pattern_section

    # ── Print to console ───────────────────────────────────────────────
    print(full_report)

    # ── Save text report ───────────────────────────────────────────────
    txt_path = output_dir / f"{model_name}_failure_report.txt"
    txt_path.write_text(full_report, encoding="utf-8")
    print(f"\nText report saved → {txt_path}")

    # ── Save structured JSON ───────────────────────────────────────────
    json_out = {
        "model": model_name,
        "dataset_consistency_score": scored["dataset_consistency_score"],
        "num_families": scored["num_families"],
        "worst_families": worst,
        "intervention_failure_stats": {
            k: {kk: vv for kk, vv in v.items() if kk != "examples"}
            for k, v in intervention_stats.items()
        },
        "intervention_failure_examples": {
            k: v["examples"] for k, v in intervention_stats.items()
        },
    }
    json_path = output_dir / f"{model_name}_failure_cases.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(json_out, f, indent=2, ensure_ascii=False)
    print(f"Structured JSON saved → {json_path}")


if __name__ == "__main__":
    main()
