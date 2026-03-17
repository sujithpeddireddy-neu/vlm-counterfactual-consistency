import json
from pathlib import Path
from typing import Dict, List, Tuple


def normalize_text(value) -> str:
    if value is None:
        return ""
    return str(value).strip().lower()


def load_json(json_path: str):
    with Path(json_path).open("r", encoding="utf-8") as f:
        return json.load(f)


def answers_are_different(a: str, b: str) -> bool:
    return normalize_text(a) != normalize_text(b)


def contradiction_check(original_pred: str, cf_pred: str) -> Tuple[bool, str]:
    orig = normalize_text(original_pred)
    cf = normalize_text(cf_pred)

    if orig in {"yes", "no"} and cf in {"yes", "no"}:
        return (orig != cf, f"expected opposite yes/no answer; original={orig}, counterfactual={cf}")

    return (False, f"unsupported contradiction pair: original={orig}, counterfactual={cf}")


def entailment_check(cf_pred: str) -> Tuple[bool, str]:
    cf = normalize_text(cf_pred)
    return (cf == "yes", f"entailed question should be yes; got {cf}")


def attribute_change_check(original_pred: str, cf_pred: str, expected_answer: str) -> Tuple[bool, str]:
    orig = normalize_text(original_pred)
    cf = normalize_text(cf_pred)
    expected = normalize_text(expected_answer)

    if expected:
        return (cf == expected, f"expected swapped attribute '{expected}'; got '{cf}'")

    return (answers_are_different(orig, cf), f"counterfactual attribute should differ from original; original={orig}, counterfactual={cf}")


def object_change_check(original_pred: str, cf_pred: str, expected_answer: str) -> Tuple[bool, str]:
    orig = normalize_text(original_pred)
    cf = normalize_text(cf_pred)
    expected = normalize_text(expected_answer)

    if expected:
        return (cf == expected, f"expected swapped object '{expected}'; got '{cf}'")

    return (answers_are_different(orig, cf), f"counterfactual object should differ from original; original={orig}, counterfactual={cf}")


def spatial_change_check(original_pred: str, cf_pred: str) -> Tuple[bool, str]:
    orig = normalize_text(original_pred)
    cf = normalize_text(cf_pred)

    if orig in {"yes", "no"} and cf in {"yes", "no"}:
        return (orig != cf, f"spatial yes/no answer should usually change; original={orig}, counterfactual={cf}")

    return (answers_are_different(orig, cf), f"spatial answer should differ from original; original={orig}, counterfactual={cf}")


def score_counterfactual(original_prediction: str, cf_item: Dict) -> Dict:
    relation = cf_item.get("logical_relation", "")
    cf_prediction = cf_item.get("model_prediction", "")
    expected_answer = cf_item.get("expected_answer", "")

    if relation == "contradiction":
        passed, reason = contradiction_check(original_prediction, cf_prediction)
    elif relation == "entails":
        passed, reason = entailment_check(cf_prediction)
    elif relation == "attribute_change":
        passed, reason = attribute_change_check(original_prediction, cf_prediction, expected_answer)
    elif relation == "object_change":
        passed, reason = object_change_check(original_prediction, cf_prediction, expected_answer)
    elif relation == "spatial_change":
        passed, reason = spatial_change_check(original_prediction, cf_prediction)
    else:
        passed, reason = False, f"unknown logical relation: {relation}"

    return {
        "intervention_type": cf_item.get("intervention_type"),
        "logical_relation": relation,
        "counterfactual_question": cf_item.get("counterfactual_question"),
        "expected_answer": expected_answer,
        "model_prediction": cf_prediction,
        "passed": passed,
        "reason": reason,
    }


def score_family(prediction_family: Dict) -> Dict:
    original = prediction_family.get("original", {})
    original_prediction = original.get("model_prediction", "")
    counterfactuals = prediction_family.get("counterfactuals", [])

    scored_items = [score_counterfactual(original_prediction, item) for item in counterfactuals]

    passed_count = sum(1 for item in scored_items if item["passed"])
    total = len(scored_items)
    family_score = passed_count / total if total > 0 else 0.0

    return {
        "question_id": prediction_family.get("question_id"),
        "image_id": prediction_family.get("image_id"),
        "original_question": original.get("question"),
        "original_answer": original.get("answer"),
        "original_model_prediction": original_prediction,
        "passed_count": passed_count,
        "total_counterfactuals": total,
        "family_consistency_score": family_score,
        "details": scored_items,
    }


def score_dataset(prediction_families: List[Dict]) -> Dict:
    family_results = [score_family(family) for family in prediction_families]
    total_score = sum(item["family_consistency_score"] for item in family_results)
    dataset_score = total_score / len(family_results) if family_results else 0.0

    return {
        "num_families": len(family_results),
        "dataset_consistency_score": dataset_score,
        "family_results": family_results,
    }


def save_json(obj, output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def main() -> None:
    input_path = "results/predictions/mock_predictions.json"
    output_path = "results/metrics/mock_consistency_scores.json"

    prediction_families = load_json(input_path)
    scored = score_dataset(prediction_families)
    save_json(scored, output_path)

    print(f"Scored {scored['num_families']} families")
    print(f"Dataset Consistency Score: {scored['dataset_consistency_score']:.4f}")
    print(f"Saved results to: {output_path}")


if __name__ == "__main__":
    main()