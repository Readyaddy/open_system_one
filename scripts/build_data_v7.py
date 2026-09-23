"""Materializes dataset_v7's combined intent corpus and the QQP
paraphrase-pair auxiliary data to disk under data/, so they're inspectable
and versioned rather than only ever existing as an in-memory HF datasets
cache. Run this once (or whenever dataset_v7.py / paraphrase_aux.py
change) to refresh data/.
"""
import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dataset_v7 import build_combined_dataset
from paraphrase_aux import load_qqp_pairs

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    print("Building combined intent corpus (dataset_v7)...")
    data = build_combined_dataset()

    intent_dir = os.path.join(DATA_DIR, "intent_corpus")
    os.makedirs(intent_dir, exist_ok=True)

    for split_name in ["train", "val", "test", "test_oos", "test_zero_shot"]:
        rows = [{"text": t, "label": l} for t, l in data[split_name]]
        write_jsonl(os.path.join(intent_dir, f"{split_name}.jsonl"), rows)
        print(f"  {split_name}.jsonl: {len(rows)} rows")

    with open(os.path.join(intent_dir, "labels.json"), "w", encoding="utf-8") as f:
        json.dump({
            "seen_labels": data["seen_labels"],
            "all_labels": data["all_labels"],
            "zero_shot_labels": data["zero_shot_labels"],
        }, f, indent=2)
    print(f"  labels.json: {len(data['seen_labels'])} seen, {len(data['all_labels'])} total, "
          f"{len(data['zero_shot_labels'])} zero-shot")

    print("\nBuilding QQP paraphrase-pair auxiliary data...")
    qqp_dir = os.path.join(DATA_DIR, "qqp_paraphrase_pairs")
    os.makedirs(qqp_dir, exist_ok=True)
    qqp_train, qqp_val = load_qqp_pairs(max_train=100000, max_val=10000)

    write_jsonl(os.path.join(qqp_dir, "train.jsonl"),
                [{"text1": a, "text2": b, "is_paraphrase": p} for a, b, p in qqp_train])
    write_jsonl(os.path.join(qqp_dir, "val.jsonl"),
                [{"text1": a, "text2": b, "is_paraphrase": p} for a, b, p in qqp_val])
    print(f"  train.jsonl: {len(qqp_train)} pairs")
    print(f"  val.jsonl: {len(qqp_val)} pairs")

    print("\nDone.")


if __name__ == "__main__":
    main()
