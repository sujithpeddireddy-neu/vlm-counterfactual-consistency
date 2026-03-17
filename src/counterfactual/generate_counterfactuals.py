import json
from pathlib import Path
from typing import Dict, List


INTERVENTION_TYPES = [
    "negation",
    "attribute_swap",
    "entailment",
    "spatial_perturbation",
]


def load_questions(json_path: str) -> List[Dict]:
    path = Path(json_path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def generate_negation(example: Dict) -> Dict:
    q = example["question"]
    a = example["answer"]

    if a.lower() in {"yes", "no"}:
        flipped = "no" if a.lower() == "yes" else "yes"
        cf_question = q
        expected = flipped
    else:
        cf_question = f"Is it not true that {q[:-1].lower()}?" if q.endswith("?") else f"Is it not true that {q.lower()}?"
        expected = "unknown"

    return {
        "intervention_type": "negation",
        "counterfactual_question": cf_question,
        "expected_answer": expected,
        "logical_relation": "negation",
    }


def generate_attribute_swap(example: Dict) -> Dict:
    q = example["question"]
    a = example["answer"]

    swap_map = {
        "red": "blue",
        "blue": "red",
        "black": "white",
        "white": "black",
        "small": "large",
        "large": "small",
    }

    swapped = swap_map.get(a.lower(), f"not_{a.lower()}")
    return {
        "intervention_type": "attribute_swap",
        "counterfactual_question": q,
        "expected_answer": swapped,
        "logical_relation": "attribute_change",
    }


def generate_entailment(example: Dict) -> Dict:
    qtype = example.get("question_type", "").lower()
    a = example["answer"]

    if qtype == "attribute":
        cf_question = f"Is there an object with attribute {a}?"
        expected = "yes"
    elif qtype == "object":
        cf_question = f"Is the person holding something?"
        expected = "yes"
    elif qtype == "spatial":
        cf_question = "Is the object visible in the image?"
        expected = "yes"
    else:
        cf_question = "Does the image contain the relevant object(s)?"
        expected = "yes"

    return {
        "intervention_type": "entailment",
        "counterfactual_question": cf_question,
        "expected_answer": expected,
        "logical_relation": "entails",
    }


def generate_spatial_perturbation(example: Dict) -> Dict:
    q = example["question"]

    replacements = {
        " on ": " under ",
        " under ": " on ",
        " left of ": " right of ",
        " right of ": " left of ",
        " in front of ": " behind ",
        " behind ": " in front of ",
    }

    new_q = q
    changed = False
    for old, new in replacements.items():
        if old in f" {q.lower()} ":
            new_q = q.lower().replace(old.strip(), new.strip())
            new_q = new_q[0].upper() + new_q[1:]
            changed = True
            break

    if not changed:
        new_q = f"[spatial perturbation needed] {q}"

    return {
        "intervention_type": "spatial_perturbation",
        "counterfactual_question": new_q,
        "expected_answer": "unknown",
        "logical_relation": "spatial_change",
    }


def generate_counterfactual_family(example: Dict) -> Dict:
    family = {
        "question_id": example["question_id"],
        "image_id": example["image_id"],
        "original": {
            "question": example["question"],
            "answer": example["answer"],
            "question_type": example.get("question_type", "unknown"),
        },
        "counterfactuals": [
            generate_negation(example),
            generate_attribute_swap(example),
            generate_entailment(example),
            generate_spatial_perturbation(example),
        ],
    }
    return family


def save_counterfactuals(families: List[Dict], output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(families, f, indent=2, ensure_ascii=False)


def main():
    input_path = "data/raw/gqa_sample/questions.json"
    output_path = "data/counterfactual/gqa_sample_counterfactuals.json"

    questions = load_questions(input_path)
    families = [generate_counterfactual_family(q) for q in questions]
    save_counterfactuals(families, output_path)

    print(f"Loaded {len(questions)} questions")
    print(f"Saved counterfactual families to: {output_path}")


if __name__ == "__main__":
    main()