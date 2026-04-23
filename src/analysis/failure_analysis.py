# failure_analysis.py - extract and display failure cases from consistency scoring results
import argparse
import json
from collections import defaultdict
from pathlib import Path


def load_scored(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def infer_model_name(path):
    return Path(path).stem.split("_consistency")[0]


def worst_families(scored, top_n):
    families = scored.get("family_results", [])
    return sorted(families, key=lambda f: f["family_consistency_score"])[:top_n]


def failure_summary_by_intervention(scored):
    stats = defaultdict(lambda: {"total": 0, "failed": 0, "examples": []})
    for family in scored.get("family_results", []):
        orig_q = family.get("original_question", "")
        for detail in family.get("details", []):
            itype = detail["intervention_type"]
            stats[itype]["total"] += 1
            if not detail["passed"]:
                stats[itype]["failed"] += 1
                if len(stats[itype]["examples"]) < 3:
                    stats[itype]["examples"].append({
                        "original_question": orig_q,
                        "counterfactual_question": detail.get("counterfactual_question", ""),
                        "expected_answer": detail.get("expected_answer", ""),
                        "model_prediction": detail.get("model_prediction", ""),
                        "reason": detail.get("reason", ""),
                    })

    for itype, counts in stats.items():
        counts["failure_rate"] = round(counts["failed"] / counts["total"], 4) if counts["total"] else 0.0
    return dict(sorted(stats.items(), key=lambda x: -x[1]["failure_rate"]))


def _truncate(s, n=90):
    return s if len(s) <= n else s[:n - 3] + "..."


def format_family(rank, family):
    lines = []
    score = family["family_consistency_score"]
    passed = family["passed_count"]
    total = family["total_counterfactuals"]
    qtype = family.get("question_type", "unknown")

    lines.append(f"#{rank} [{qtype}] Score: {score:.2f} ({passed}/{total} passed) id={family.get('question_id', '')}")
    lines.append(f"Q: {_truncate(family.get('original_question', ''))}")
    lines.append(f"GT: {family.get('original_answer', '')} | Predicted: {_truncate(str(family.get('original_model_prediction', '')), 40)}")

    for cf_num, detail in enumerate(family.get("details", []), 1):
        icon = "[PASS]" if detail["passed"] else "[FAIL]"
        lines.append(f"  CF{cf_num} [{detail['intervention_type']}] {icon}")
        lines.append(f"    Q: {_truncate(detail.get('counterfactual_question', ''))}")
        lines.append(f"    Expected: {detail.get('expected_answer', '')}")
        lines.append(f"    Got: {_truncate(str(detail.get('model_prediction', '')), 60)}")
        if not detail["passed"]:
            lines.append(f"    Reason: {_truncate(detail.get('reason', ''), 70)}")

    lines.append("")
    return "\n".join(lines)


def format_intervention_summary(stats):
    lines = ["", "Failure patterns by intervention type", "-" * 40]
    for itype, data in stats.items():
        pct = data["failure_rate"] * 100
        lines.append(f"\n{itype}: {data['failed']}/{data['total']} failed ({pct:.1f}%)")
        for ex in data["examples"]:
            lines.append(f"  example:")
            lines.append(f"    orig: {_truncate(ex['original_question'], 70)}")
            lines.append(f"    cf: {_truncate(ex['counterfactual_question'], 70)}")
            lines.append(f"    expected: {ex['expected_answer']}")
            lines.append(f"    got: {_truncate(str(ex['model_prediction']), 60)}")
    return "\n".join(lines)


def format_header(model_name, scored, top_n):
    return (
        f"\nFailure case analysis - {model_name}\n"
        f"Total families: {scored['num_families']}\n"
        f"Dataset score: {scored['dataset_consistency_score']:.4f}\n"
        f"Showing worst {top_n} families\n"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Extract and report failure cases from scored consistency results."
    )
    parser.add_argument("--results", required=True,
                        help="*_consistency_scores.json from run_benchmark.py")
    parser.add_argument("--top-n", type=int, default=20, dest="top_n",
                        help="Number of worst families to display")
    parser.add_argument("--output", default="results/analysis/")
    args = parser.parse_args()

    scored = load_scored(args.results)
    model_name = infer_model_name(args.results)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    header = format_header(model_name, scored, args.top_n)

    worst = worst_families(scored, args.top_n)
    family_blocks = "\n".join(
        format_family(rank + 1, fam) for rank, fam in enumerate(worst)
    )

    intervention_stats = failure_summary_by_intervention(scored)
    pattern_section = format_intervention_summary(intervention_stats)

    full_report = header + "\n" + family_blocks + "\n" + pattern_section

    print(full_report)

    txt_path = output_dir / f"{model_name}_failure_report.txt"
    txt_path.write_text(full_report, encoding="utf-8")
    print(f"\nText report saved -> {txt_path}")

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
    print(f"Structured JSON saved -> {json_path}")


if __name__ == "__main__":
    main()
