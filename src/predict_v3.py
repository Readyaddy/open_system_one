"""Demo: load the v3 checkpoint (mpnet backbone, CLINC150+Banking77, 227
intents) and route new utterances by embedding compatibility.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import torch
from dataset_v3 import build_combined_dataset, base_description, raw_intent_name
from model_v3 import JEPADecisionModelV3, get_tokenizer


def load(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    model = JEPADecisionModelV3().to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Loaded checkpoint from epoch {ckpt['epoch']} (val_acc {ckpt['val_acc']:.4f})")
    return model


def predict(model, tokenizer, device, text, labels, outcome_texts, top_k=3):
    tok = tokenizer([text], padding=True, truncation=True, max_length=32, return_tensors="pt")
    out_tok = tokenizer(outcome_texts, padding=True, truncation=True, max_length=32, return_tensors="pt")
    with torch.no_grad():
        ctx_emb = model.encode_context(tok["input_ids"].to(device), tok["attention_mask"].to(device))
        out_emb = model.encode_outcome(out_tok["input_ids"].to(device), out_tok["attention_mask"].to(device))
        sims = (ctx_emb @ out_emb.t()).squeeze(0)
        probs = torch.softmax(sims / 0.07, dim=-1)
    ranked = sorted(zip(labels, probs.tolist()), key=lambda x: -x[1])
    return ranked[:top_k]


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = build_combined_dataset()
    labels = data["labels"]
    outcome_texts = [base_description(raw_intent_name(l)) for l in labels]

    tokenizer = get_tokenizer()
    ckpt_path = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "jepa_v3_best.pt")
    model = load(ckpt_path, device)

    samples = [
        "hey what's it like outside right now",
        "can you set a wake up call for 6am",
        "book me a room in chicago for next friday",
        "i still haven't gotten my new card in the mail",
        "someone else is using my card, i didn't make these purchases",
        "the atm ate my card and didn't give it back",
        "why was my payment declined",
        "asdlkj random keyboard mashing not a real request",
    ]
    for s in samples:
        top3 = predict(model, tokenizer, device, s, labels, outcome_texts)
        print(f"\n{s!r}")
        for label, p in top3:
            print(f"   {label:35s} {p:.3f}")
