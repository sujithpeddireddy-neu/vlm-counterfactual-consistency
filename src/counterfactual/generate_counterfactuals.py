import json
import re
from pathlib import Path
from typing import Dict, List, Optional


COLOR_SWAP = {
    "red": "blue",
    "blue": "red",
    "black": "white",
    "white": "black",
    "green": "yellow",
    "yellow": "green",
    "brown": "gray",
    "gray": "brown",
    "pink": "purple",
    "purple": "pink",
    "orange": "black",
}

SIZE_SWAP = {
    "small": "large",
    "large": "small",
    "big": "small",
    "tall": "short",
    "short": "tall",
}

MATERIAL_SWAP = {
    "wood": "metal",
    "metal": "wood",
    "plastic": "glass",
    "glass": "plastic",
}

OBJECT_SWAP = {
    "umbrella": "bag",
    "bag": "umbrella",
    "ball": "frisbee",
    "frisbee": "ball",
    "phone": "book",
    "book": "phone",
    "cup": "bottle",
    "bottle": "cup",
}

SPATIAL_SWAP = {
    "on": "under",
    "under": "on",
    "left of": "right of",
    "right of": "left of",
    "in front of": "behind",
    "behind": "in front of",
    "above": "below",
    "below": "above",
}

YES_NO_STARTERS = (
    "is ",
    "are ",
    "does ",
    "do ",
    "did ",
    "can ",
    "could ",
    "will ",
    "would ",
    "has ",
    "have ",
    "had ",
)


def load_questions(json_path: str) -> List[Dict]:
    with Path(json_path).open("r", encoding="utf-8") as f:
        return json.load(f)


def save_counterfactuals(families: List[Dict], output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(families, f, indent=2, ensure_ascii=False)


def normalize_question(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def is_yes_no_question(question: str) -> bool:
    q = question.strip().lower()
    return q.startswith(YES_NO_STARTERS)


def strip_qmark(question: str) -> str:
    return question[:-1] if question.endswith("?") else question


def swap_from_map(answer: str, mapping: Dict[str, str]) -> Optional[str]:
    return mapping.get(answer.lower())


_CONTRACTION_MAP = {
    "is":    "Isn't",
    "are":   "Aren't",
    "does":  "Doesn't",
    "do":    "Don't",
    "did":   "Didn't",
    "can":   "Can't",
    "could": "Couldn't",
    "will":  "Won't",
    "would": "Wouldn't",
    "has":   "Hasn't",
    "have":  "Haven't",
    "had":   "Hadn't",
}


def _negate_yesno(q: str) -> Optional[str]:
    """
    Turn 'Is the car red?' → 'Isn't the car red?' using contraction substitution.
    Returns None if no leading auxiliary verb is recognised.
    """
    for verb, contracted in _CONTRACTION_MAP.items():
        pattern = rf"^{verb}\b"
        if re.match(pattern, q.strip(), re.IGNORECASE):
            negated = re.sub(pattern, contracted, q.strip(), count=1, flags=re.IGNORECASE)
            if not negated.endswith("?"):
                negated += "?"
            return negated
    return None


def build_negation(example: Dict) -> Dict:
    q = normalize_question(example["question"])
    a = str(example["answer"]).strip().lower()

    if is_yes_no_question(q) and a in {"yes", "no"}:
        negated_q = _negate_yesno(q)
        if negated_q is not None:
            return {
                "intervention_type": "negation",
                "counterfactual_question": negated_q,
                "expected_answer": "no" if a == "yes" else "yes",
                "logical_relation": "contradiction",
                "generation_note": f"contraction negation: '{q}' → '{negated_q}'",
            }

    # Fallback for non-yes/no questions or unrecognised auxiliary verbs
    lowered = strip_qmark(q).lower()
    return {
        "intervention_type": "negation",
        "counterfactual_question": f"Is it false that {lowered}?",
        "expected_answer": "yes",
        "logical_relation": "negation",
        "generation_note": "fallback negation template",
    }


def _swap_in_question(q: str, old: str, new: str) -> Optional[str]:
    """Replace first word-boundary occurrence of `old` in `q` (case-insensitive).
    Returns the modified question, or None if `old` is not found."""
    pattern = rf"\b{re.escape(old.lower())}\b"
    if not re.search(pattern, q.lower()):
        return None
    replaced = re.sub(pattern, new, q.lower(), count=1)
    replaced = replaced[0].upper() + replaced[1:]
    if not replaced.endswith("?"):
        replaced += "?"
    return replaced


def build_attribute_swap(example: Dict) -> Dict:
    q = normalize_question(example["question"])
    a = str(example["answer"]).strip()
    a_lower = a.lower()

    cf_question: Optional[str] = None
    expected: Optional[str] = None
    note: str = ""

    # ── Case 1: yes/no question — scan question text for a swappable attribute ──
    # e.g. "Is the car red?" → "Is the car blue?", expected flips yes↔no
    if is_yes_no_question(q) and a_lower in {"yes", "no"}:
        for mapping in (COLOR_SWAP, SIZE_SWAP, MATERIAL_SWAP):
            for attr_val, swapped_val in mapping.items():
                swapped_q = _swap_in_question(q, attr_val, swapped_val)
                if swapped_q is not None:
                    cf_question = swapped_q
                    expected = "no" if a_lower == "yes" else "yes"
                    note = f"swapped '{attr_val}' → '{swapped_val}' in question; answer flipped"
                    break
            if cf_question:
                break

    # ── Case 2: open-ended — try replacing the answer value in the question ──
    # e.g. "What is the color of the red car?" + answer "red" → swap "red" in question
    if cf_question is None:
        swapped_ans = (
            swap_from_map(a, COLOR_SWAP)
            or swap_from_map(a, SIZE_SWAP)
            or swap_from_map(a, MATERIAL_SWAP)
        )
        if swapped_ans:
            swapped_q = _swap_in_question(q, a, swapped_ans)
            if swapped_q is not None:
                cf_question = swapped_q
                expected = swapped_ans
                note = f"swapped answer value '{a_lower}' → '{swapped_ans}' in question"
            else:
                # Answer not in question text (typical open-ended): keep question, swap expected
                cf_question = q
                expected = swapped_ans
                note = f"attribute answer swapped to '{swapped_ans}' (value absent from question)"
        else:
            cf_question = q
            expected = f"not_{a_lower}"
            note = "no swap mapping found; negated expected answer"

    return {
        "intervention_type": "attribute_swap",
        "counterfactual_question": cf_question,
        "expected_answer": expected,
        "logical_relation": "attribute_change",
        "generation_note": note,
    }


def build_object_swap(example: Dict) -> Dict:
    q = normalize_question(example["question"])
    a = str(example["answer"]).strip()

    swapped = swap_from_map(a, OBJECT_SWAP) or f"different_{a.lower()}"

    # Try to replace the object name in the question text
    # e.g. "Is the person holding an umbrella?" → "Is the person holding a bag?"
    swapped_q = _swap_in_question(q, a, swapped)
    if swapped_q is not None:
        cf_question = swapped_q
        note = f"object '{a.lower()}' → '{swapped}' swapped in question text"
    else:
        cf_question = q
        note = f"object answer swapped to '{swapped}' (object absent from question)"

    return {
        "intervention_type": "object_swap",
        "counterfactual_question": cf_question,
        "expected_answer": swapped,
        "logical_relation": "object_change",
        "generation_note": note,
    }


def build_entailment(example: Dict) -> Dict:
    qtype = str(example.get("question_type", "")).lower()
    a = str(example["answer"]).strip().lower()

    if qtype == "attribute":
        # Only use answer as attribute label when it's a real attribute value (not yes/no).
        if a not in {"yes", "no", ""}:
            cf_question = f"Is there something that has the attribute '{a}'?"
        else:
            cf_question = "Are the attributes of the objects in the image visible?"
    elif qtype == "object":
        cf_question = "Is the referenced object present in the image?"
    elif qtype == "spatial":
        cf_question = "Are the referenced objects visible in the image?"
    else:
        cf_question = "Is the entity or relation mentioned in the question present in the image?"

    return {
        "intervention_type": "entailment",
        "counterfactual_question": cf_question,
        "expected_answer": "yes",
        "logical_relation": "entails",
        "generation_note": "weakened statement entailed by original QA pair",
    }


def replace_spatial_phrase(question: str) -> Optional[str]:
    q = question
    lowered = q.lower()

    for old, new in sorted(SPATIAL_SWAP.items(), key=lambda x: len(x[0]), reverse=True):
        pattern = rf"\b{re.escape(old)}\b"
        if re.search(pattern, lowered):
            replaced = re.sub(pattern, new, lowered, count=1)
            replaced = replaced[0].upper() + replaced[1:]
            if not replaced.endswith("?"):
                replaced += "?"
            return replaced
    return None


def build_spatial_perturbation(example: Dict) -> Dict:
    q = normalize_question(example["question"])
    replaced = replace_spatial_phrase(q)

    if replaced is not None:
        return {
            "intervention_type": "spatial_perturbation",
            "counterfactual_question": replaced,
            "expected_answer": "unknown",
            "logical_relation": "spatial_change",
            "generation_note": "spatial phrase swapped",
        }

    return {
        "intervention_type": "spatial_perturbation",
        "counterfactual_question": q,
        "expected_answer": "unknown",
        "logical_relation": "spatial_change",
        "generation_note": "no spatial phrase found; left unchanged",
    }


def select_counterfactuals(example: Dict) -> List[Dict]:
    q = normalize_question(example["question"])
    qtype = str(example.get("question_type", "")).lower()

    outputs = []

    # Yes/no questions: negation is valid; spatial perturbation only if spatial phrase exists
    if is_yes_no_question(q):
        outputs.append(build_negation(example))
        outputs.append(build_entailment(example))

        if replace_spatial_phrase(q) is not None or qtype == "spatial":
            outputs.append(build_spatial_perturbation(example))

        return outputs

    # Attribute questions: attribute swap + entailment
    if qtype == "attribute":
        outputs.append(build_attribute_swap(example))
        outputs.append(build_entailment(example))
        return outputs

    # Object questions: object swap + entailment
    if qtype == "object":
        outputs.append(build_object_swap(example))
        outputs.append(build_entailment(example))
        return outputs

    # Spatial non-yes/no questions: only include spatial perturbation if it actually changes something
    if qtype == "spatial":
        outputs.append(build_entailment(example))
        if replace_spatial_phrase(q) is not None:
            outputs.append(build_spatial_perturbation(example))
        return outputs

    # Fallback: keep only broadly valid transformations
    outputs.append(build_entailment(example))
    return outputs


def generate_counterfactual_family(example: Dict) -> Dict:
    return {
        "question_id": example["question_id"],
        "image_id": example["image_id"],
        "original": {
            "question": normalize_question(example["question"]),
            "answer": example["answer"],
            "question_type": example.get("question_type", "unknown"),
        },
        "counterfactuals": select_counterfactuals(example),
    }


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(
        description="Generate counterfactual families from internal-format questions JSON."
    )
    parser.add_argument(
        "--input", default="data/raw/gqa_sample/questions.json",
        help="Input questions JSON (list of {question_id, image_id, question, answer, question_type})",
    )
    parser.add_argument(
        "--output", default="data/counterfactual/gqa_sample_counterfactuals.json",
        help="Output path for counterfactual families JSON",
    )
    args = parser.parse_args()

    questions = load_questions(args.input)
    families = [generate_counterfactual_family(q) for q in questions]
    save_counterfactuals(families, args.output)

    print(f"Loaded {len(questions)} questions")
    print(f"Saved {len(families)} counterfactual families to: {args.output}")


if __name__ == "__main__":
    main()