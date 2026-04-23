import json
import re
from pathlib import Path


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


def load_questions(json_path):
    with Path(json_path).open("r", encoding="utf-8") as f:
        return json.load(f)


def save_counterfactuals(families, output_path):
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(families, f, indent=2, ensure_ascii=False)


def normalize_question(text):
    return re.sub(r"\s+", " ", text.strip())


def is_yes_no_question(question):
    return question.strip().lower().startswith(YES_NO_STARTERS)


def strip_qmark(question):
    return question[:-1] if question.endswith("?") else question


def swap_from_map(answer, mapping):
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


def _negate_yesno(question):
    # turn "Is the car red?" into "Isn't the car red?" using contraction substitution
    for verb, contracted in _CONTRACTION_MAP.items():
        pattern = rf"^{verb}\b"
        if re.match(pattern, question.strip(), re.IGNORECASE):
            negated = re.sub(pattern, contracted, question.strip(), count=1, flags=re.IGNORECASE)
            if not negated.endswith("?"):
                negated += "?"
            return negated
    return None


def build_negation(example):
    question = normalize_question(example["question"])
    answer = str(example["answer"]).strip().lower()

    if is_yes_no_question(question) and answer in {"yes", "no"}:
        negated = _negate_yesno(question)
        if negated is not None:
            return {
                "intervention_type": "negation",
                "counterfactual_question": negated,
                "expected_answer": "no" if answer == "yes" else "yes",
                "logical_relation": "contradiction",
                "generation_note": f"contraction negation: '{question}' -> '{negated}'",
            }

    # Fallback for non-yes/no questions or unrecognized auxiliary verbs
    base_text = strip_qmark(question).lower()
    return {
        "intervention_type": "negation",
        "counterfactual_question": f"Is it false that {base_text}?",
        "expected_answer": "yes",
        "logical_relation": "negation",
        "generation_note": "fallback negation template",
    }


def _swap_in_question(question, old_word, new_word):
    # replace first word-boundary match of old_word in question (case-insensitive)
    pattern = rf"\b{re.escape(old_word.lower())}\b"
    if not re.search(pattern, question.lower()):
        return None
    replaced = re.sub(pattern, new_word, question.lower(), count=1)
    replaced = replaced[0].upper() + replaced[1:]
    if not replaced.endswith("?"):
        replaced += "?"
    return replaced


def build_attribute_swap(example):
    question = normalize_question(example["question"])
    answer = str(example["answer"]).strip()
    answer_lower = answer.lower()

    cf_question = None
    expected = None
    note = ""

    # Case 1: yes/no question — scan question text for a swappable attribute.
    # e.g. "Is the car red?" -> "Is the car blue?", expected flips yes↔no
    if is_yes_no_question(question) and answer_lower in {"yes", "no"}:
        for mapping in (COLOR_SWAP, SIZE_SWAP, MATERIAL_SWAP):
            for attr, swapped in mapping.items():
                swapped_q = _swap_in_question(question, attr, swapped)
                if swapped_q is not None:
                    cf_question = swapped_q
                    expected = "no" if answer_lower == "yes" else "yes"
                    note = f"swapped '{attr}' -> '{swapped}' in question; answer flipped"
                    break
            if cf_question:
                break

    # Case 2: open-ended — try replacing the answer value in the question.
    # e.g. "What color is the car?" + answer "red" -> swap "red" in question
    if cf_question is None:
        swapped_answer = (
            swap_from_map(answer, COLOR_SWAP)
            or swap_from_map(answer, SIZE_SWAP)
            or swap_from_map(answer, MATERIAL_SWAP)
        )
        if swapped_answer:
            swapped_q = _swap_in_question(question, answer, swapped_answer)
            if swapped_q is not None:
                cf_question = swapped_q
                expected = swapped_answer
                note = f"swapped answer value '{answer_lower}' -> '{swapped_answer}' in question"
            else:
                # Answer not in question text (typical open-ended): keep question, swap expected
                cf_question = question
                expected = swapped_answer
                note = f"attribute answer swapped to '{swapped_answer}' (value absent from question)"
        else:
            cf_question = question
            expected = f"not_{answer_lower}"
            note = "no swap mapping found; negated expected answer"

    return {
        "intervention_type": "attribute_swap",
        "counterfactual_question": cf_question,
        "expected_answer": expected,
        "logical_relation": "attribute_change",
        "generation_note": note,
    }


def build_object_swap(example):
    question = normalize_question(example["question"])
    answer = str(example["answer"]).strip()

    swapped = swap_from_map(answer, OBJECT_SWAP) or f"different_{answer.lower()}"

    # Try to replace the object name in the question text.
    # e.g. "Is the person holding an umbrella?" -> "Is the person holding a bag?"
    swapped_q = _swap_in_question(question, answer, swapped)
    if swapped_q is not None:
        cf_question = swapped_q
        note = f"object '{answer.lower()}' -> '{swapped}' swapped in question text"
    else:
        cf_question = question
        note = f"object answer swapped to '{swapped}' (object absent from question)"

    return {
        "intervention_type": "object_swap",
        "counterfactual_question": cf_question,
        "expected_answer": swapped,
        "logical_relation": "object_change",
        "generation_note": note,
    }


def build_entailment(example):
    question_type = str(example.get("question_type", "")).lower()
    answer = str(example["answer"]).strip().lower()

    if question_type == "attribute":
        # Only use answer as attribute label when it's a real attribute value (not yes/no).
        if answer not in {"yes", "no", ""}:
            cf_question = f"Is there something that has the attribute '{answer}'?"
        else:
            cf_question = "Are the attributes of the objects in the image visible?"
    elif question_type == "object":
        cf_question = "Is the referenced object present in the image?"
    elif question_type == "spatial":
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


def replace_spatial_phrase(question):
    q_lower = question.lower()

    for phrase, replacement in sorted(SPATIAL_SWAP.items(), key=lambda x: len(x[0]), reverse=True):
        pattern = rf"\b{re.escape(phrase)}\b"
        if re.search(pattern, q_lower):
            replaced = re.sub(pattern, replacement, q_lower, count=1)
            replaced = replaced[0].upper() + replaced[1:]
            if not replaced.endswith("?"):
                replaced += "?"
            return replaced
    return None


def build_spatial_perturbation(example):
    question = normalize_question(example["question"])
    swapped_question = replace_spatial_phrase(question)

    if swapped_question is not None:
        return {
            "intervention_type": "spatial_perturbation",
            "counterfactual_question": swapped_question,
            "expected_answer": "unknown",
            "logical_relation": "spatial_change",
            "generation_note": "spatial phrase swapped",
        }

    return {
        "intervention_type": "spatial_perturbation",
        "counterfactual_question": question,
        "expected_answer": "unknown",
        "logical_relation": "spatial_change",
        "generation_note": "no spatial phrase found; left unchanged",
    }


def select_counterfactuals(example):
    question = normalize_question(example["question"])
    question_type = str(example.get("question_type", "")).lower()

    counterfactuals = []

    # Yes/no questions: negation is always valid; add spatial only if a spatial phrase exists
    if is_yes_no_question(question):
        counterfactuals.append(build_negation(example))
        counterfactuals.append(build_entailment(example))

        if replace_spatial_phrase(question) is not None or question_type == "spatial":
            counterfactuals.append(build_spatial_perturbation(example))

        return counterfactuals

    if question_type == "attribute":
        counterfactuals.append(build_attribute_swap(example))
        counterfactuals.append(build_entailment(example))
        return counterfactuals

    if question_type == "object":
        counterfactuals.append(build_object_swap(example))
        counterfactuals.append(build_entailment(example))
        return counterfactuals

    # Spatial open-ended: only include spatial perturbation if it actually changes something
    if question_type == "spatial":
        counterfactuals.append(build_entailment(example))
        if replace_spatial_phrase(question) is not None:
            counterfactuals.append(build_spatial_perturbation(example))
        return counterfactuals

    # Fallback: entailment is broadly valid for any question type
    counterfactuals.append(build_entailment(example))
    return counterfactuals


def generate_counterfactual_family(example):
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


def main():
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
