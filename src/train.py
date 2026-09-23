"""Train the JEPA-style dual-encoder decision model.

Objective: for each context, its embedding should be most compatible
(highest cosine similarity) with its correct outcome's embedding, out of
all K fixed candidate outcomes. This is InfoNCE / softmax cross-entropy
over compatibility scores -- the "logits" are similarities in embedding
space, not a classifier head tied to label identities. That's what makes
it swappable: at eval time we can re-describe a candidate outcome in new
words and it still has to work, because the outcome encoder has to
understand the description, not just recognize a memorized id.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn as nn
from dataset import build_splits, OUTCOMES, OUTCOMES_PARAPHRASED, OUTCOME_KEYS
from model import JEPADecisionModel, Vocab, pad_batch


def make_outcome_bank(vocab, device):
    ids = [vocab.encode(OUTCOMES[k]) for k in OUTCOME_KEYS]
    tok, mask = pad_batch(ids)
    return tok.to(device), mask.to(device)


def make_outcome_bank_paraphrased(vocab, device):
    ids = [vocab.encode(OUTCOMES_PARAPHRASED[k]) for k in OUTCOME_KEYS]
    tok, mask = pad_batch(ids)
    return tok.to(device), mask.to(device)


def batch_of(examples, vocab, device):
    texts = [e[0] for e in examples]
    labels = [OUTCOME_KEYS.index(e[1]) for e in examples]
    ids = [vocab.encode(t) for t in texts]
    tok, mask = pad_batch(ids)
    return tok.to(device), mask.to(device), torch.tensor(labels, dtype=torch.long, device=device)


def evaluate(model, examples, vocab, outcome_tok, outcome_mask, device):
    model.eval()
    with torch.no_grad():
        ctx_tok, ctx_mask, labels = batch_of(examples, vocab, device)
        ctx_emb = model.encode_context(ctx_tok, ctx_mask)
        out_emb = model.encode_outcome(outcome_tok, outcome_mask)
        logits = model.compatibility(ctx_emb, out_emb)
        preds = logits.argmax(dim=-1)
        acc = (preds == labels).float().mean().item()
    model.train()
    return acc


def main():
    device = torch.device("cpu")
    train_ex, val_ex, test_ex = build_splits(seed=0)

    vocab = Vocab().build([t for t, _ in train_ex] + list(OUTCOMES.values()) + list(OUTCOMES_PARAPHRASED.values()))
    print(f"Vocab size: {len(vocab)}")
    print(f"Train/val/test sizes: {len(train_ex)}/{len(val_ex)}/{len(test_ex)}")

    model = JEPADecisionModel(vocab_size=len(vocab)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    ce = nn.CrossEntropyLoss()

    outcome_tok, outcome_mask = make_outcome_bank(vocab, device)
    outcome_tok_para, outcome_mask_para = make_outcome_bank_paraphrased(vocab, device)

    batch_size = 32
    epochs = 40
    n = len(train_ex)

    for epoch in range(1, epochs + 1):
        perm = torch.randperm(n).tolist()
        total_loss = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            batch = [train_ex[j] for j in idx]
            ctx_tok, ctx_mask, labels = batch_of(batch, vocab, device)

            ctx_emb = model.encode_context(ctx_tok, ctx_mask)
            out_emb = model.encode_outcome(outcome_tok, outcome_mask)  # (K, D), recompute each step (small K)
            logits = model.compatibility(ctx_emb, out_emb)  # (B, K)

            loss = ce(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(idx)

        if epoch % 5 == 0 or epoch == 1:
            val_acc = evaluate(model, val_ex, vocab, outcome_tok, outcome_mask, device)
            val_acc_para = evaluate(model, val_ex, vocab, outcome_tok_para, outcome_mask_para, device)
            print(f"epoch {epoch:3d}  loss {total_loss / n:.4f}  val_acc {val_acc:.3f}  "
                  f"val_acc(paraphrased outcomes) {val_acc_para:.3f}")

    test_acc = evaluate(model, test_ex, vocab, outcome_tok, outcome_mask, device)
    test_acc_para = evaluate(model, test_ex, vocab, outcome_tok_para, outcome_mask_para, device)
    print("\n=== Final ===")
    print(f"test_acc (trained outcome descriptions):     {test_acc:.3f}")
    print(f"test_acc (paraphrased, unseen descriptions):  {test_acc_para:.3f}")

    os.makedirs(os.path.join(os.path.dirname(__file__), "..", "checkpoints"), exist_ok=True)
    ckpt_path = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "jepa_decision_model.pt")
    torch.save({
        "model_state": model.state_dict(),
        "vocab": vocab.tok2idx,
    }, ckpt_path)
    print(f"Saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()
