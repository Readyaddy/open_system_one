"""Builds the hard reasoning corpus for Experiment 7 (exp7_hard_reasoning_corpus).

Downloads and formats 4 high-difficulty reasoning datasets:
  1. ARC-Challenge (allenai/ai2_arc) -- multi-choice physical/causal science.
  2. ProofWriter (tasksource/proofwriter) -- systematic multi-step rule chaining.
  3. MuSR (TAUR-Lab/MuSR) -- narrative state tracking (murder mysteries, object placements, team allocation).
  4. WMDP (cais/wmdp) -- advanced domain multi-choice reasoning (bio/cyber/chem).

All examples are converted into Open System-1 schema:
  {
    "context": str,
    "instructions": str,
    "option_texts": List[str],
    "qtype": "choice" | "bool",
    "answer_idx": int,
    "source": str
  }

Saved to data/exp7_hard_reasoning_corpus/{train,val,test}.jsonl
"""
import json
import os
import random

from datasets import load_dataset

random.seed(20260923)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
OUT_DIR = os.path.join(DATA_DIR, "exp7_hard_reasoning_corpus")


def _write_jsonl(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  wrote {len(rows)} rows -> {path}", flush=True)


def build_arc(max_train=10000):
    print("  building ARC-Challenge & ARC-Easy...")
    rows = {"train": [], "val": [], "test": []}
    for config in ["ARC-Challenge", "ARC-Easy"]:
        ds = load_dataset("allenai/ai2_arc", config)
        for split_name, hf_split in [("train", "train"), ("val", "validation"), ("test", "test")]:
            if hf_split not in ds:
                continue
            for ex in ds[hf_split]:
                labels = ex["choices"]["label"]
                texts = ex["choices"]["text"]
                ans_key = ex["answerKey"]
                
                if ans_key in labels:
                    ans_idx = labels.index(ans_key)
                elif ans_key.isdigit() and int(ans_key) - 1 < len(texts):
                    ans_idx = int(ans_key) - 1
                elif ans_key in ["1", "2", "3", "4"] and int(ans_key) - 1 < len(texts):
                    ans_idx = int(ans_key) - 1
                else:
                    continue

                rows[split_name].append({
                    "context": "",
                    "instructions": ex["question"],
                    "option_texts": texts,
                    "qtype": "choice",
                    "answer_idx": ans_idx,
                    "source": f"arc_{config.lower()}"
                })
    return rows


def build_proofwriter(max_train=25000):
    print("  building ProofWriter (rule-chaining logic)...")
    rows = {"train": [], "val": [], "test": []}
    ds = load_dataset("tasksource/proofwriter")
    for split_name, hf_split in [("train", "train"), ("val", "validation"), ("test", "test")]:
        if hf_split not in ds:
            continue
        data = ds[hf_split]
        idx = random.sample(range(len(data)), min(max_train, len(data))) if split_name == "train" else range(min(5000, len(data)))
        for i in idx:
            ex = data[i]
            ans_str = str(ex["answer"]).strip().lower()
            if ans_str not in ["true", "false", "unknown"]:
                continue
            
            options = ["True", "False", "Unknown"]
            ans_idx = options.index(ans_str.capitalize())

            rows[split_name].append({
                "context": ex["theory"],
                "instructions": f"Based on the theory, is the following statement True, False, or Unknown: \"{ex['question']}\"?",
                "option_texts": options,
                "qtype": "choice",
                "answer_idx": ans_idx,
                "source": "proofwriter"
            })
    return rows


def build_musr():
    print("  building MuSR (narrative constraint tracking)...")
    rows = {"train": [], "val": [], "test": []}
    musr = load_dataset("TAUR-Lab/MuSR", "default")
    
    all_examples = []
    for split_key in ["murder_mysteries", "object_placements", "team_allocation"]:
        for ex in musr[split_key]:
            narrative = ex.get("narrative", "")
            question = ex.get("question", "")
            choices = ex.get("choices", [])
            ans_idx = ex.get("answer_index", 0)
            
            if not choices or ans_idx >= len(choices):
                continue

            all_examples.append({
                "context": narrative,
                "instructions": question,
                "option_texts": choices,
                "qtype": "choice",
                "answer_idx": ans_idx,
                "source": f"musr_{split_key}"
            })
    
    random.seed(42)
    random.shuffle(all_examples)
    n = len(all_examples)
    n_train = int(n * 0.7)
    n_val = int(n * 0.15)

    rows["train"] = all_examples[:n_train]
    rows["val"] = all_examples[n_train:n_train + n_val]
    rows["test"] = all_examples[n_train + n_val:]
    return rows


def build_wmdp():
    print("  building WMDP (bio/cyber/chem domain decision questions)...")
    rows = {"train": [], "val": [], "test": []}
    all_examples = []
    for config in ["wmdp-bio", "wmdp-cyber", "wmdp-chem"]:
        try:
            ds = load_dataset("cais/wmdp", config)
            for ex in ds["test"]:
                choices = ex["choices"]
                ans_idx = ex["answer"]
                if ans_idx >= len(choices):
                    continue
                all_examples.append({
                    "context": "",
                    "instructions": ex["question"],
                    "option_texts": choices,
                    "qtype": "choice",
                    "answer_idx": ans_idx,
                    "source": config
                })
        except Exception as e:
            print(f"    warning: failed to load {config}: {e}")

    random.seed(123)
    random.shuffle(all_examples)
    n = len(all_examples)
    n_train = int(n * 0.7)
    n_val = int(n * 0.15)

    rows["train"] = all_examples[:n_train]
    rows["val"] = all_examples[n_train:n_train + n_val]
    rows["test"] = all_examples[n_train + n_val:]
    return rows


def main():
    print("=== Building Hard Reasoning Corpus (exp7_hard_reasoning_corpus) ===")
    all_rows = {"train": [], "val": [], "test": []}

    for builder in [build_arc, build_proofwriter, build_musr, build_wmdp]:
        part = builder()
        for split in ["train", "val", "test"]:
            all_rows[split].extend(part[split])

    for split in ["train", "val", "test"]:
        random.seed(20260923)
        random.shuffle(all_rows[split])
        _write_jsonl(all_rows[split], os.path.join(OUT_DIR, f"{split}.jsonl"))

    print("\n=== Corpus Summary ===")
    for split in ["train", "val", "test"]:
        print(f"  {split}: {len(all_rows[split])} examples")


if __name__ == "__main__":
    main()
