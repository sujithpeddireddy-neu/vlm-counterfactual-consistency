# gqa_loader.py - convert GQA questions JSON to the pipeline internal format
import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


# Type mapping
def map_question_type(types):
    structural = types.get("structural", "")
    semantic   = types.get("semantic",   "")
    if structural == "verify": return "yes/no"
    if semantic   == "attr":   return "attribute"
    if semantic   == "rel":    return "spatial"
    if semantic   == "obj":    return "object"
    return structural or "unknown"


# Loading and filtering
def load_gqa(questions_path, images_dir, max_samples=None, keep_types=None, seed=42):
    # load GQA questions, filter to existing images, map types, optionally sample
    images_dir = Path(images_dir)

    print(f"Loading {questions_path} ...")
    with open(questions_path, encoding="utf-8") as f:
        raw = json.load(f)
    print(f"  {len(raw):,} questions in file")

    # Convert and filter in one pass
    records = []
    missing_images = 0
    for qid, item in raw.items():
        image_id = item["imageId"]
        if not (images_dir / f"{image_id}.jpg").exists():
            missing_images += 1
            continue
        qtype = map_question_type(item.get("types", {}))
        if keep_types and qtype not in keep_types:
            continue
        records.append(
            {
                "question_id":   qid,
                "image_id":      image_id,
                "question":      item["question"],
                "answer":        item["answer"],
                "question_type": qtype,
            }
        )

    if missing_images:
        print(f"  Skipped {missing_images:,} questions (image not found)")

    # Optional type filter report
    if keep_types:
        print(f"  After type filter ({keep_types}): {len(records):,} questions")

    # Stratified sampling
    if max_samples and len(records) > max_samples:
        records = _stratified_sample(records, max_samples, seed)

    return records


def _stratified_sample(records, n, seed):
    # sample n records proportionally from each question_type bucket
    rng = random.Random(seed)
    by_type = defaultdict(list)
    for record in records:
        by_type[record["question_type"]].append(record)

    total = len(records)
    sampled = []
    remainder = []

    for qtype, items in by_type.items():
        quota = max(1, round(n * len(items) / total))
        rng.shuffle(items)
        sampled.extend(items[:quota])
        remainder.extend(items[quota:])

    # top up or trim to exactly n
    if len(sampled) < n:
        rng.shuffle(remainder)
        sampled.extend(remainder[: n - len(sampled)])
    sampled = sampled[:n]
    rng.shuffle(sampled)
    return sampled


# Stats display
def print_stats(records):
    type_counts = Counter(r["question_type"] for r in records)
    print(f"\n  Total questions selected: {len(records):,}")
    print("  Question type distribution:")
    for qtype, count in type_counts.most_common():
        pct = count / len(records) * 100
        print(f"    {qtype:<20} {count:>6,}  ({pct:.1f}%)")


# Entry point
def main():
    parser = argparse.ArgumentParser(
        description="Convert GQA questions JSON to the pipeline's internal format."
    )
    parser.add_argument(
        "--questions", required=True,
        help="GQA questions JSON file (e.g. Data/questions/val_balanced_questions.json)",
    )
    parser.add_argument(
        "--images", required=True,
        help="Directory containing {imageId}.jpg files (e.g. Data/images)",
    )
    parser.add_argument(
        "--output", required=True,
        help="Output path for internal-format questions JSON",
    )
    parser.add_argument(
        "--max-samples", type=int, default=None, dest="max_samples",
        help="Stratified sample down to this many questions (default: use all)",
    )
    parser.add_argument(
        "--question-types", nargs="+", default=None, dest="question_types",
        metavar="TYPE",
        help="Only keep these question types (e.g. yes/no attribute spatial object)",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    records = load_gqa(
        questions_path=args.questions,
        images_dir=args.images,
        max_samples=args.max_samples,
        keep_types=args.question_types,
        seed=args.seed,
    )

    print_stats(records)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved -> {out}")


if __name__ == "__main__":
    main()
