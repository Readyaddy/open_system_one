"""Checks that S1DecisionModel IS exp7a: strict load + reproduce the exact
end-of-epoch eval exp7a recorded in its checkpoint (epoch 10:
val_acc 0.6433, zero_shot_acc 0.7133, banking77_holdout_acc 0.5200).

exp7's train.py evaluated with data.build_intent_example(n_target=50) under
random.Random(9000/9500/9900 + epoch) on the first 300/300/400 examples. The
same calls are replayed here through exp7's own data.py, so the examples are
identical; only the model code differs. Both head_input variants are tried.
"""
import importlib.util
import os
import random
import sys

import torch
from torch.amp import autocast

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
import s1_model as SM  # noqa: E402
import s1_data as SD  # noqa: E402

# exp7's data.py does `from model import PackedExample`; point that at exp7's model.
sys.modules["model"] = SM.E7
EXP7 = os.path.join(HERE, "..", "exp7_hybrid_decision")
spec = importlib.util.spec_from_file_location("exp7_data", os.path.join(EXP7, "data.py"))
D7 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(D7)

CKPT = os.path.join(HERE, "..", "..", "exp7_best_zeroshot.pt")


@torch.no_grad()
def acc(model, packer, exs, dev):
    model.eval()
    ok = n = 0
    pairs = [(e, SD.Meta("x", "x", e.qtype, len(e.option_texts))) for e in exs]
    for mb in packer.pack(pairs):
        packed, opt = mb.to(dev)
        with autocast("cuda", dtype=torch.bfloat16):
            logits = model(packed, opt)
        ok += (logits.argmax(-1) == packed.answer_idx).sum().item()
        n += len(mb.metas)
    return ok / n


def main():
    dev = torch.device("cuda")
    tok = SM.get_tokenizer(SM.BACKBONE)
    builder = SM.PackedSequenceBuilder(tok, budget_total=2048, l_context=768, l_instructions=96, l_max_per_option=64)
    packer = SD.Packer(builder, tok, max_tokens=16000, max_options=1024)
    intent = D7.load_intent_corpus_minus_banking77()
    epoch = 10

    def examples(rows, pool, seed):
        rng = random.Random(seed)
        return [D7.build_intent_example(t, l, pool, rng, n_target=50) for t, l in rows]

    sets = {
        "val_acc (0.6433)": examples(intent.val[:300], intent.seen_labels, 9000 + epoch),
        "zero_shot_acc (0.7133)": examples(intent.test_zero_shot[:300], intent.all_labels, 9500 + epoch),
        "banking77 (0.5200)": examples(intent.banking77_holdout[:400], intent.banking77_labels, 9900 + epoch),
    }
    model = SM.S1DecisionModel(mask_token_id=tok.mask_token_id, gradient_checkpointing=False)
    ck, missing, unexpected = model.load_exp7a(CKPT, strict=True)
    print(f"strict load OK: missing={len(missing)} unexpected={len(unexpected)}  "
          f"(ckpt epoch {ck['epoch']}, inject_gate {model.inject_gate.item():.4f})", flush=True)
    model.to(dev)
    for variant in ("s0x2", "s0"):
        model.head_input = variant
        res = {k: acc(model, packer, v, dev) for k, v in sets.items()}
        print(f"  head_input={variant:5s}  " + "  ".join(f"{k}: {v:.4f}" for k, v in res.items()), flush=True)


if __name__ == "__main__":
    main()
