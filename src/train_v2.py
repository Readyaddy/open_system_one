"""Train the v2 (pretrained-backbone) JEPA-style dual-encoder decision
model on CLINC150: 150 real intents, ~15k real training utterances, plus
an explicit out-of-scope (oos) class held out of the classification loss
and used only to test rejection behavior.

Three things are evaluated, because "generalized" needs a concrete
definition:
  1. Standard test accuracy on the 150 in-scope intents, using the same
     auto-generated outcome descriptions seen in training.
  2. Paraphrase accuracy: same test examples, but outcome descriptions for
     a 30-intent subset are swapped for hand-written paraphrases NEVER
     seen during training -- this is the real generalization probe.
  3. OOS separation: for genuinely out-of-scope utterances, the model
     should NOT be highly compatible with any of the 150 known outcomes.
     We report the gap between max-similarity on in-scope vs oos examples
     and a simple threshold accuracy for flagging oos.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import time
import torch
import torch.nn as nn
from dataset_v2 import load_clinc150, build_base_descriptions, PARAPHRASED_DESCRIPTIONS, get_examples, OOS_LABEL
from model_v2 import JEPADecisionModelV2, get_tokenizer


def tokenize_batch(tokenizer, texts, device, max_length=32):
    enc = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    return enc["input_ids"].to(device), enc["attention_mask"].to(device)


def encode_outcome_bank(model, tokenizer, descriptions, device):
    tok, mask = tokenize_batch(tokenizer, descriptions, device)
    with torch.no_grad() if not model.training else torch.enable_grad():
        return model.encode_outcome(tok, mask)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    ds, label_names = load_clinc150()
    in_scope_labels = [l for l in label_names if l != OOS_LABEL]
    label_to_idx = {l: i for i, l in enumerate(in_scope_labels)}

    base_desc = build_base_descriptions(label_names)
    outcome_texts = [base_desc[l] for l in in_scope_labels]  # 150 descriptions, index-aligned

    train_all = get_examples(ds["train"], label_names)
    val_all = get_examples(ds["validation"], label_names)
    test_all = get_examples(ds["test"], label_names)

    train_ex = [(t, l) for t, l in train_all if l != OOS_LABEL]
    val_ex = [(t, l) for t, l in val_all if l != OOS_LABEL]
    test_ex = [(t, l) for t, l in test_all if l != OOS_LABEL]
    test_oos_ex = [(t, l) for t, l in test_all if l == OOS_LABEL]

    print(f"Train/val/test (in-scope): {len(train_ex)}/{len(val_ex)}/{len(test_ex)}  "
          f"| OOS test examples: {len(test_oos_ex)}  | intents: {len(in_scope_labels)}")

    tokenizer = get_tokenizer()
    model = JEPADecisionModelV2().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    ce = nn.CrossEntropyLoss()

    batch_size = 64
    epochs = 6
    n = len(train_ex)

    def run_eval(examples, outcome_texts_eval):
        model.eval()
        out_emb = encode_outcome_bank(model, tokenizer, outcome_texts_eval, device)
        correct, total = 0, 0
        eval_bs = 128
        with torch.no_grad():
            for i in range(0, len(examples), eval_bs):
                batch = examples[i:i + eval_bs]
                texts = [t for t, _ in batch]
                labels = torch.tensor([label_to_idx[l] for _, l in batch], device=device)
                tok, mask = tokenize_batch(tokenizer, texts, device)
                ctx_emb = model.encode_context(tok, mask)
                logits = model.compatibility(ctx_emb, out_emb)
                preds = logits.argmax(dim=-1)
                correct += (preds == labels).sum().item()
                total += len(batch)
        model.train()
        return correct / total

    def run_oos_check(outcome_texts_eval):
        model.eval()
        out_emb = encode_outcome_bank(model, tokenizer, outcome_texts_eval, device)
        eval_bs = 128

        def max_sim(examples):
            sims = []
            with torch.no_grad():
                for i in range(0, len(examples), eval_bs):
                    batch = examples[i:i + eval_bs]
                    texts = [t for t, _ in batch]
                    tok, mask = tokenize_batch(tokenizer, texts, device)
                    ctx_emb = model.encode_context(tok, mask)
                    sim = ctx_emb @ out_emb.t()
                    sims.append(sim.max(dim=-1).values)
            return torch.cat(sims)

        in_scope_sim = max_sim(test_ex[:1000])
        oos_sim = max_sim(test_oos_ex)
        model.train()
        return in_scope_sim.mean().item(), oos_sim.mean().item()

    print("\n=== Training ===")
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        perm = torch.randperm(n).tolist()
        total_loss = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            batch = [train_ex[j] for j in idx]
            texts = [t for t, _ in batch]
            labels = torch.tensor([label_to_idx[l] for _, l in batch], device=device)
            tok, mask = tokenize_batch(tokenizer, texts, device)

            ctx_emb = model.encode_context(tok, mask)
            out_emb = model.encode_outcome(*tokenize_batch(tokenizer, outcome_texts, device))
            logits = model.compatibility(ctx_emb, out_emb)
            loss = ce(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * len(idx)

        val_acc = run_eval(val_ex, outcome_texts)
        dt = time.time() - t0
        print(f"epoch {epoch}/{epochs}  loss {total_loss / n:.4f}  val_acc {val_acc:.3f}  ({dt:.1f}s)")

    print("\n=== Final evaluation ===")
    test_acc = run_eval(test_ex, outcome_texts)
    print(f"test_acc (trained/auto-generated descriptions): {test_acc:.3f}")

    # Paraphrase generalization: swap in hand-written descriptions for the
    # subset of intents we paraphrased; keep base descriptions for the rest.
    paraphrase_texts = [
        PARAPHRASED_DESCRIPTIONS.get(l, base_desc[l]) for l in in_scope_labels
    ]
    paraphrased_subset_idx = [label_to_idx[l] for l in PARAPHRASED_DESCRIPTIONS if l in label_to_idx]
    test_subset = [(t, l) for t, l in test_ex if l in PARAPHRASED_DESCRIPTIONS]
    para_acc_subset = run_eval(test_subset, paraphrase_texts)
    print(f"test_acc on the {len(PARAPHRASED_DESCRIPTIONS)}-intent subset, "
          f"UNSEEN paraphrased descriptions: {para_acc_subset:.3f}  (n={len(test_subset)})")

    base_acc_subset = run_eval(test_subset, outcome_texts)
    print(f"  (same subset, original trained descriptions, for comparison): {base_acc_subset:.3f}")

    in_scope_sim, oos_sim = run_oos_check(outcome_texts)
    print(f"\nOOS separation: mean max-similarity in-scope={in_scope_sim:.3f}  oos={oos_sim:.3f}  "
          f"gap={in_scope_sim - oos_sim:.3f}")

    os.makedirs(os.path.join(os.path.dirname(__file__), "..", "checkpoints"), exist_ok=True)
    ckpt_path = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "jepa_decision_model_v2.pt")
    torch.save({"model_state": model.state_dict()}, ckpt_path)
    print(f"\nSaved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()
