"""Standalone final-evaluation runner for a saved v5 checkpoint.

train_v5.py's background process was killed (session ended) partway
through epoch 7, after saving a valid best checkpoint at epoch 6
(val_intent_acc 0.9412). Rather than re-running ~3 hours of training,
this loads that checkpoint directly and runs the exact same final
evaluation block train_v5.py would have run: seen-intent test accuracy,
paraphrase generalization, true zero-shot accuracy, OOS separation, and
the joint-vs-separate-passes timing benchmark.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import argparse
import time
import torch
from torch.amp import autocast

from dataset_v5 import build_multi_question_dataset
from dataset_v4 import base_description, raw_intent_name
from paraphrases_v4 import get_paraphrases
from model_v5 import MultiQuestionPolyEncoderV5, get_tokenizer, QUESTION_TYPES, QTYPE_TO_ID


def tokenize(tokenizer, texts, device, max_length=32):
    enc = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    return enc["input_ids"].to(device), enc["attention_mask"].to(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="jepa_v5_best.pt")
    parser.add_argument("--freeze_layers", type=int, default=16)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    data = build_multi_question_dataset()
    test_ex, test_oos_ex, test_zs_ex = data["test"], data["test_oos"], data["test_zero_shot"]
    seen_labels, all_labels = data["seen_labels"], data["all_labels"]
    zero_shot_labels = set(data["zero_shot_labels"])
    seen_idx = {l: i for i, l in enumerate(seen_labels)}
    all_idx = {l: i for i, l in enumerate(all_labels)}

    tokenizer = get_tokenizer()
    model = MultiQuestionPolyEncoderV5(freeze_layers=args.freeze_layers).to(device)

    ckpt_path = os.path.join(os.path.dirname(__file__), "..", "checkpoints", args.ckpt)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Loaded checkpoint from epoch {ckpt['epoch']} (val_intent_acc {ckpt['val_score']:.4f})", flush=True)

    base_seen_texts = [base_description(raw_intent_name(l)) for l in seen_labels]
    base_all_texts = [base_description(raw_intent_name(l)) for l in all_labels]

    INTENT_QID = QTYPE_TO_ID["intent"]
    NH_QID = QTYPE_TO_ID["needs_human"]
    URG_QID = QTYPE_TO_ID["urgency"]

    def run_eval(examples, outcome_texts_eval, idx_map, eval_bs=64):
        otok, omask = tokenize(tokenizer, outcome_texts_eval, device)
        with torch.no_grad():
            out_emb = model.encode_outcome(otok, omask)

        intent_correct = nh_correct = urg_correct = 0
        urg_abs_err = 0
        total = 0
        with torch.no_grad():
            for i in range(0, len(examples), eval_bs):
                batch = examples[i:i + eval_bs]
                texts = [e["text"] for e in batch]
                intent_lab = torch.tensor([idx_map[e["intent_label"]] for e in batch], device=device)
                nh_lab = torch.tensor([float(e["needs_human_label"]) for e in batch], device=device)
                urg_lab = torch.tensor([e["urgency_label"] - 1 for e in batch], device=device)

                ctok, cmask = tokenize(tokenizer, texts, device)
                with autocast(device_type=device.type, enabled=(device.type == "cuda")):
                    shared = model.encode_context_all(ctok, cmask)
                    intent_logits = model.intent_logits(shared[:, INTENT_QID], out_emb)
                    nh_logits = model.needs_human_logits(shared[:, NH_QID])
                    urg_logits = model.urgency_logits(shared[:, URG_QID])

                intent_preds = intent_logits.argmax(dim=-1)
                nh_preds = (torch.sigmoid(nh_logits) > 0.5).float()
                urg_preds = urg_logits.argmax(dim=-1)

                intent_correct += (intent_preds == intent_lab).sum().item()
                nh_correct += (nh_preds == nh_lab).sum().item()
                urg_correct += (urg_preds == urg_lab).sum().item()
                urg_abs_err += (urg_preds - urg_lab).abs().sum().item()
                total += len(batch)

        return {
            "intent_acc": intent_correct / total,
            "needs_human_acc": nh_correct / total,
            "urgency_acc": urg_correct / total,
            "urgency_mae": urg_abs_err / total,
        }

    def run_oos_check(outcome_texts_eval, n_sample=1000, eval_bs=64):
        otok, omask = tokenize(tokenizer, outcome_texts_eval, device)
        with torch.no_grad():
            out_emb = model.encode_outcome(otok, omask)

        def max_logit(examples):
            sims = []
            with torch.no_grad():
                for i in range(0, len(examples), eval_bs):
                    batch = examples[i:i + eval_bs]
                    texts = [e["text"] for e in batch]
                    ctok, cmask = tokenize(tokenizer, texts, device)
                    with autocast(device_type=device.type, enabled=(device.type == "cuda")):
                        shared = model.encode_context_all(ctok, cmask)
                        logits = model.intent_logits(shared[:, INTENT_QID], out_emb)
                    sims.append(logits.max(dim=-1).values)
            return torch.cat(sims)

        in_scope = max_logit(test_ex[:n_sample])
        oos = max_logit(test_oos_ex)
        return in_scope.mean().item(), oos.mean().item()

    def run_timing_benchmark(n_examples=256, eval_bs=32):
        examples = test_ex[:n_examples]
        texts_batches = [[e["text"] for e in examples[i:i + eval_bs]] for i in range(0, len(examples), eval_bs)]
        tok_batches = [tokenize(tokenizer, tb, device) for tb in texts_batches]

        with torch.no_grad():
            for ctok, cmask in tok_batches[:2]:
                with autocast(device_type=device.type, enabled=(device.type == "cuda")):
                    model.encode_context_all(ctok, cmask)
        if device.type == "cuda":
            torch.cuda.synchronize()

        t0 = time.time()
        with torch.no_grad():
            for ctok, cmask in tok_batches:
                with autocast(device_type=device.type, enabled=(device.type == "cuda")):
                    model.context_encoder(ctok, cmask, torch.arange(3, device=device))
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_joint = time.time() - t0

        t0 = time.time()
        with torch.no_grad():
            for ctok, cmask in tok_batches:
                for qid in range(3):
                    with autocast(device_type=device.type, enabled=(device.type == "cuda")):
                        model.context_encoder(ctok, cmask, torch.tensor([qid], device=device))
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_separate = time.time() - t0

        return t_joint, t_separate

    print("\n=== Final evaluation (from saved checkpoint) ===", flush=True)
    test_metrics = run_eval(test_ex, base_seen_texts, seen_idx)
    test_nh_pos_rate = sum(e["needs_human_label"] for e in test_ex) / len(test_ex)
    test_nh_majority = max(test_nh_pos_rate, 1 - test_nh_pos_rate)
    test_urg_counts = {i: sum(1 for e in test_ex if e["urgency_label"] == i) for i in range(1, 6)}
    test_urg_majority = max(test_urg_counts.values()) / len(test_ex)
    print(f"test (seen intents, base descriptions): intent_acc={test_metrics['intent_acc']:.4f}  "
          f"needs_human_acc={test_metrics['needs_human_acc']:.4f} (majority-baseline={test_nh_majority:.4f})  "
          f"urgency_acc={test_metrics['urgency_acc']:.4f} (majority-baseline={test_urg_majority:.4f})  "
          f"urgency_mae={test_metrics['urgency_mae']:.3f}", flush=True)

    paraphrases = get_paraphrases()
    paraphrases = {k: v for k, v in paraphrases.items() if k in seen_idx}
    para_texts = [paraphrases.get(l, base_description(raw_intent_name(l))) for l in seen_labels]
    test_para_subset = [e for e in test_ex if e["intent_label"] in paraphrases]
    para_metrics = run_eval(test_para_subset, para_texts, seen_idx)
    base_metrics_subset = run_eval(test_para_subset, base_seen_texts, seen_idx)
    print(f"\ntest, {len(paraphrases)}-intent subset, UNSEEN paraphrases: "
          f"intent_acc={para_metrics['intent_acc']:.4f}  (n={len(test_para_subset)})", flush=True)
    print(f"  same subset, base descriptions: intent_acc={base_metrics_subset['intent_acc']:.4f}", flush=True)

    zs_metrics = run_eval(test_zs_ex, base_all_texts, all_idx)
    seen_mixed_metrics = run_eval(test_ex[:1500], base_all_texts, all_idx)
    chance = 1.0 / len(all_labels)
    print(f"\nZERO-SHOT test ({len(zero_shot_labels)} never-trained intents, "
          f"candidate pool = all {len(all_labels)} intents): "
          f"intent_acc={zs_metrics['intent_acc']:.4f}  (n={len(test_zs_ex)})", flush=True)
    print(f"  (reference) seen-intent acc in same mixed pool: {seen_mixed_metrics['intent_acc']:.4f}  "
          f"| chance: {chance:.4f}", flush=True)

    in_scope_sim, oos_sim = run_oos_check(base_seen_texts)
    print(f"\nOOS separation: mean max-compat-logit  in-scope={in_scope_sim:.4f}  oos={oos_sim:.4f}  "
          f"gap={in_scope_sim - oos_sim:.4f}", flush=True)

    t_joint, t_separate = run_timing_benchmark()
    print(f"\nTiming (256 test contexts, {len(QUESTION_TYPES)} question types): "
          f"joint single-pass={t_joint:.3f}s  separate {len(QUESTION_TYPES)}x-pass={t_separate:.3f}s  "
          f"speedup={t_separate / t_joint:.2f}x", flush=True)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
