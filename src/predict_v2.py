"""Demo: load the v2 (pretrained-backbone) model and route new utterances
by embedding compatibility across all 150 CLINC150 intents, plus a couple
of totally new outcome descriptions written by hand (not intent names from
the dataset) to show the outcome side generalizes to fresh text.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import torch
from dataset_v2 import load_clinc150, build_base_descriptions, OOS_LABEL
from model_v2 import JEPADecisionModelV2, get_tokenizer


def load(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    model = JEPADecisionModelV2().to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def predict(model, tokenizer, device, text, outcome_labels, outcome_texts):
    tok = tokenizer([text], padding=True, truncation=True, max_length=32, return_tensors="pt")
    out_tok = tokenizer(outcome_texts, padding=True, truncation=True, max_length=32, return_tensors="pt")
    with torch.no_grad():
        ctx_emb = model.encode_context(tok["input_ids"].to(device), tok["attention_mask"].to(device))
        out_emb = model.encode_outcome(out_tok["input_ids"].to(device), out_tok["attention_mask"].to(device))
        sims = (ctx_emb @ out_emb.t()).squeeze(0)
        probs = torch.softmax(sims / 0.07, dim=-1)
    ranked = sorted(zip(outcome_labels, probs.tolist()), key=lambda x: -x[1])
    return ranked[:3]


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, label_names = load_clinc150()
    in_scope_labels = [l for l in label_names if l != OOS_LABEL]
    base_desc = build_base_descriptions(label_names)
    outcome_texts = [base_desc[l] for l in in_scope_labels]

    tokenizer = get_tokenizer()
    ckpt_path = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "jepa_decision_model_v2.pt")
    model = load(ckpt_path, device)

    samples = [
        "hey what's it like outside right now",
        "can you set a wake up call for 6am",
        "i need to know how much is in my checking account",
        "book me a room in chicago for next friday",
        "asdlkj random keyboard mashing not a real request",
    ]
    for s in samples:
        top3 = predict(model, tokenizer, device, s, in_scope_labels, outcome_texts)
        print(f"\n{s!r}")
        for label, p in top3:
            print(f"   {label:30s} {p:.3f}")
