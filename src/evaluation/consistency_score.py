from typing import List, Dict


def compute_consistency_score(prediction_family: List[Dict]) -> float:
    """
    Placeholder consistency score:
    returns 1.0 if no pair is explicitly marked contradictory, else 0.0.

    Later this should be replaced with real logic for:
    - negation consistency
    - entailment consistency
    - spatial relation consistency
    - attribute swap compatibility
    """
    for item in prediction_family:
        if item.get("is_contradictory", False):
            return 0.0
    return 1.0


if __name__ == "__main__":
    demo = [
        {"question": "Is the car red?", "prediction": "yes", "is_contradictory": False},
        {"question": "Is the car blue?", "prediction": "no", "is_contradictory": False},
    ]
    print("Consistency Score:", compute_consistency_score(demo))