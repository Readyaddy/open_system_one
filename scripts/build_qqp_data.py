"""Rebuilds ONLY data/qqp_paraphrase_pairs/ at a larger size, without
touching intent_corpus/ or mcq_corpus/ -- meant to be run directly on a
Colab VM (downloads from HuggingFace there, avoiding a slow local upload
of the larger resulting files), same pattern as build_mcq_data.py.
"""
import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from paraphrase_aux import load_qqp_pairs, class_balance

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    max_train = int(sys.argv[1]) if len(sys.argv) > 1 else 100000
    max_val = int(sys.argv[2]) if len(sys.argv) > 2 else 10000
    print(f"Building QQP paraphrase-pair data: max_train={max_train} max_val={max_val}...")
    qqp_train, qqp_val = load_qqp_pairs(max_train=max_train, max_val=max_val)

    qqp_dir = os.path.join(DATA_DIR, "qqp_paraphrase_pairs")
    os.makedirs(qqp_dir, exist_ok=True)
    write_jsonl(os.path.join(qqp_dir, "train.jsonl"),
                [{"text1": a, "text2": b, "is_paraphrase": p} for a, b, p in qqp_train])
    write_jsonl(os.path.join(qqp_dir, "val.jsonl"),
                [{"text1": a, "text2": b, "is_paraphrase": p} for a, b, p in qqp_val])
    print(f"  train.jsonl: {len(qqp_train)} pairs, {100*class_balance(qqp_train):.1f}% positive")
    print(f"  val.jsonl: {len(qqp_val)} pairs, {100*class_balance(qqp_val):.1f}% positive")
    print("Done.")


if __name__ == "__main__":
    main()
