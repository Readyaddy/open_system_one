"""Materializes mcq_dataset.py's combined MCQ corpus to disk under
data/mcq_corpus/, same pattern as build_data_v7.py. Run once (or whenever
mcq_dataset.py changes).
"""
import sys
import os
import json
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mcq_dataset import build_mcq_dataset

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    print("Building combined MCQ corpus (mcq_dataset)...")
    data = build_mcq_dataset()

    mcq_dir = os.path.join(DATA_DIR, "mcq_corpus")
    os.makedirs(mcq_dir, exist_ok=True)

    for split_name in ["train", "val", "test"]:
        rows = data[split_name]
        write_jsonl(os.path.join(mcq_dir, f"{split_name}.jsonl"), rows)
        by_source = Counter(r["source"] for r in rows)
        print(f"  {split_name}.jsonl: {len(rows)} rows  {dict(by_source)}")

    print("\nDone.")


if __name__ == "__main__":
    main()
