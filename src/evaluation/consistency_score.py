import json
from pathlib import Path


def normalize_text(value):
    if value is None:
        return ""
    return str(value).strip().lower()


def extract_yesno(text):
    # extract bare yes/no from a verbose prediction like "Yes, the car is red."
    normalized = normalize_text(text)
    if normalized.startswith("yes"):
        return "yes"
    if normalized.startswith("no"):
        return "no"
    return normalized


def load_json(json_path):
    with Path(json_path).open("r", encoding="utf-8") as f:
        return json.load(f)


def answers_are_different(left, right):
    return normalize_text(left) != normalize_text(right)


def contradiction_check(original_pred, cf_pred):
    orig = extract_yesno(original_pred)
    cf = extract_yesno(cf_pred)

    if orig in {"yes", "no"} and cf in {"yes", "no"}:
        return (orig != cf, f"expected opposite yes/no answer; original={orig}, counterfactual={cf}")

    return (False, f"unsupported contradiction pair: original={orig}, counterfactual={cf}")


def entailment_check(cf_pred):
    cf = extract_yesno(cf_pred)
    return (cf == "yes", f"entailed question should be yes; got {cf}")


def _contains_word(text, word):
    # true if word appears as a whole word in text (case-insensitive)
    import re
    return bool(re.search(rf"\b{re.escape(word)}\b", text))


def attribute_change_check(original_pred, cf_pred, expected_answer):
    expected = normalize_text(expected_answer)
    # When the expected answer is yes/no, extract the yes/no signal from verbose predictions
    orig = extract_yesno(original_pred) if expected in {"yes", "no"} else normalize_text(original_pred)
    cf   = extract_yesno(cf_pred)       if expected in {"yes", "no"} else normalize_text(cf_pred)

    if expected:
        # Accept verbose predictions that contain the expected keyword as a whole word
        # e.g. "The fence is made of metal." should match expected="metal"
        passed = (cf == expected) or _contains_word(cf, expected)
        return (passed, f"expected swapped attribute '{expected}'; got '{cf}'")

    return (answers_are_different(orig, cf), f"counterfactual attribute should differ from original; original={orig}, counterfactual={cf}")


def object_change_check(original_pred, cf_pred, expected_answer):
    expected = normalize_text(expected_answer)
    orig = normalize_text(original_pred)
    cf   = normalize_text(cf_pred)

    if expected:
        # Accept verbose predictions that contain the expected keyword as a whole word
        passed = (cf == expected) or _contains_word(cf, expected)
        return (passed, f"expected swapped object '{expected}'; got '{cf}'")

    return (answers_are_different(orig, cf), f"counterfactual object should differ from original; original={orig}, counterfactual={cf}")


def spatial_change_check(original_pred, cf_pred):
    orig = extract_yesno(original_pred)
    cf = extract_yesno(cf_pred)

    if orig in {"yes", "no"} and cf in {"yes", "no"}:
        return (orig != cf, f"spatial yes/no answer should usually change; original={orig}, counterfactual={cf}")

    return (answers_are_different(orig, cf), f"spatial answer should differ from original; original={orig}, counterfactual={cf}")


def score_counterfactual(original_prediction, cf_item):
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


def score_family(prediction_family):
    original = prediction_family.get("original", {})
    original_prediction = original.get("model_prediction", "")
    ground_truth = original.get("answer", "")
    counterfactuals = prediction_family.get("counterfactuals", [])

    scored_items = [score_counterfactual(original_prediction, item) for item in counterfactuals]

    passed_count = sum(1 for item in scored_items if item["passed"])
    total = len(scored_items)
    family_score = passed_count / total if total > 0 else 0.0

    # Standard VQA accuracy: did the model answer the original question correctly?
    # For yes/no ground truths, extract just the yes/no signal from verbose predictions.
    gt_norm = normalize_text(ground_truth)
    pred_norm = extract_yesno(original_prediction) if gt_norm in {"yes", "no"} else normalize_text(original_prediction)
    original_correct = pred_norm == gt_norm

    return {
        "question_id": prediction_family.get("question_id"),
        "image_id": prediction_family.get("image_id"),
        "original_question": original.get("question"),
        "original_answer": ground_truth,
        "original_model_prediction": original_prediction,
        "original_correct": original_correct,
        "passed_count": passed_count,
        "total_counterfactuals": total,
        "family_consistency_score": family_score,
        "details": scored_items,
    }


def score_dataset(prediction_families):
    family_results = [score_family(family) for family in prediction_families]
    num_families = len(family_results)
    dataset_score = sum(r["family_consistency_score"] for r in family_results) / num_families if num_families else 0.0
    vqa_accuracy = sum(1 for r in family_results if r["original_correct"]) / num_families if num_families else 0.0

    return {
        "num_families": num_families,
        "dataset_consistency_score": dataset_score,
        "vqa_accuracy": round(vqa_accuracy, 4),
        "family_results": family_results,
    }


def save_json(obj, output_path):
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def main():
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