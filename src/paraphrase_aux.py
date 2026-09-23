"""Auxiliary paraphrase-pair training signal, using QQP (Quora Question
Pairs) -- real, human-written paraphrases and non-paraphrases, completely
independent of dataset_v7.py's intent data and of the 8 synthetic
templates used to augment outcome descriptions there.

WHY THIS EXISTS: the templated augmentation (dataset_v7.sample_outcome_
description) is what closed the paraphrase gap back in iteration 3, but
its entire diversity comes from 8 fixed sentence templates plus a small
synonym dictionary -- a model could learn to be invariant to THOSE 8
patterns specifically without learning genuine paraphrase invariance.
QQP pairs are real people's writing, phrased in ways no template or
synonym list would produce, so training the SAME encoders to recognize
QQP paraphrases as similar (and non-paraphrases as dissimilar) is a
direct, independent check/signal on whether the model is learning actual
semantic invariance -- not just a proxy through the intent task.

Usage: mix batches of (text1, text2, is_paraphrase) alongside the main
intent-classification batches during training, with a contrastive/binary
objective: paraphrase pairs should have HIGH compatibility, non-paraphrase
pairs LOW -- computed through the exact same context/outcome encoders as
the main task, so anything learned here directly affects the same
embedding space the intent task uses.
"""
import random
from datasets import load_dataset


def load_qqp_pairs(max_train: int = 20000, max_val: int = 2000, seed: int = 99):
    """Returns (train_pairs, val_pairs), each a list of (text1, text2,
    is_paraphrase: bool). Subsampled from QQP's 363k/40k full splits --
    we don't need the full size, just real diversity, and a smaller
    subsample keeps the auxiliary task's per-epoch cost proportionate to
    the main task rather than dwarfing it (QQP alone is >10x the size of
    the combined intent corpus)."""
    ds = load_dataset("SetFit/qqp")
    rng = random.Random(seed)

    train_rows = list(ds["train"])
    rng.shuffle(train_rows)
    train_rows = train_rows[:max_train]

    val_rows = list(ds["validation"])
    rng.shuffle(val_rows)
    val_rows = val_rows[:max_val]

    to_pairs = lambda rows: [(r["text1"], r["text2"], bool(r["label"])) for r in rows]
    return to_pairs(train_rows), to_pairs(val_rows)


def class_balance(pairs):
    pos = sum(1 for _, _, is_para in pairs if is_para)
    return pos / len(pairs) if pairs else 0.0
