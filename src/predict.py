"""Demo: load the trained model and classify new tickets by embedding
compatibility -- one forward pass each side, no generation, no decoding.

Run: python src/predict.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import torch
from dataset import OUTCOMES, OUTCOME_KEYS
from model import JEPADecisionModel, Vocab, pad_batch


def load(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    vocab = Vocab()
    vocab.tok2idx = ckpt["vocab"]
    model = JEPADecisionModel(vocab_size=len(vocab))
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, vocab


def predict(model, vocab, text, outcome_descriptions):
    keys = list(outcome_descriptions.keys())
    ctx_ids = [vocab.encode(text)]
    ctx_tok, ctx_mask = pad_batch(ctx_ids)

    out_ids = [vocab.encode(outcome_descriptions[k]) for k in keys]
    out_tok, out_mask = pad_batch(out_ids)

    with torch.no_grad():
        ctx_emb = model.encode_context(ctx_tok, ctx_mask)
        out_emb = model.encode_outcome(out_tok, out_mask)
        logits = model.compatibility(ctx_emb, out_emb)
        probs = torch.softmax(logits, dim=-1).squeeze(0)

    ranked = sorted(zip(keys, probs.tolist()), key=lambda x: -x[1])
    return ranked


if __name__ == "__main__":
    ckpt_path = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "jepa_decision_model.pt")
    model, vocab = load(ckpt_path)

    samples = [
        "My headphones showed up broken, I want my money back.",
        "URGENT: my account was charged twice, get me a manager right now.",
        "How do I change the email on my account?",
        "The app crashes every time I open settings.",
        "buy cheap watches now click here",
        "You guys are amazing, my order arrived perfect and support was so kind!",
        # A brand-new candidate outcome, never seen at all during training,
        # to show the outcome encoder is doing real semantic work.
    ]

    print("=== Predictions using the ORIGINAL trained outcome descriptions ===")
    for s in samples:
        ranked = predict(model, vocab, s, OUTCOMES)
        top = ranked[0]
        print(f"[{top[0]:>10s}] ({top[1]:.2f})  {s}")
