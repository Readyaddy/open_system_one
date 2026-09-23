"""Experiment 2 dataset: reformats the CLINC150+Banking77+SNIPS pool into
small per-example candidate sets (correct answer + sampled distractors)
instead of scoring every example against the full 199-234 label pool.
See NOTES.md in this folder for the full rationale.

Scoped to the Choice/intent task only (no needs_human/urgency questions)
-- keeping the comparison to iteration 4 limited to ONE variable, the
candidate-set framing, not also introducing iteration 5's multi-question
setup at the same time.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import random

from dataset_v4 import build_combined_dataset, base_description, raw_intent_name, sample_outcome_description


def _dataset_of(composite_label: str) -> str:
    return composite_label.split("::", 1)[0]


def make_request(text, correct_label, label_pool, label_by_dataset, rng: random.Random,
                  n_options: int = 6, augment: bool = True):
    correct_ds = _dataset_of(correct_label)

    n_hard = min(n_options - 1, max(1, (n_options - 1) * 2 // 3))
    same_ds_pool = [l for l in label_by_dataset.get(correct_ds, []) if l != correct_label]
    hard_negatives = rng.sample(same_ds_pool, min(n_hard, len(same_ds_pool)))

    remaining = n_options - 1 - len(hard_negatives)
    other_pool = [l for l in label_pool if l != correct_label and l not in hard_negatives]
    random_negatives = rng.sample(other_pool, min(remaining, len(other_pool)))

    option_labels = hard_negatives + random_negatives + [correct_label]
    rng.shuffle(option_labels)
    correct_idx = option_labels.index(correct_label)

    def desc(label):
        name = raw_intent_name(label)
        return sample_outcome_description(name, rng) if augment else base_description(name)

    option_texts = [desc(l) for l in option_labels]
    return {"text": text, "option_texts": option_texts, "correct_idx": correct_idx,
            "option_labels": option_labels, "correct_label": correct_label}


def build_request_dataset(n_options: int = 6, seed: int = 321):
    data = build_combined_dataset()
    seen_labels, all_labels = data["seen_labels"], data["all_labels"]
    zero_shot_labels = set(data["zero_shot_labels"])

    by_ds_seen, by_ds_all = {}, {}
    for l in seen_labels:
        by_ds_seen.setdefault(_dataset_of(l), []).append(l)
    for l in all_labels:
        by_ds_all.setdefault(_dataset_of(l), []).append(l)

    rng = random.Random(seed)

    def build(examples, label_pool, by_ds, augment):
        return [make_request(t, l, label_pool, by_ds, rng, n_options, augment) for t, l in examples]

    train_req = build(data["train"], seen_labels, by_ds_seen, augment=True)
    val_req = build(data["val"], seen_labels, by_ds_seen, augment=False)
    test_req = build(data["test"], seen_labels, by_ds_seen, augment=False)
    # zero-shot: distractor pool spans ALL labels (seen + never-trained),
    # so a never-trained intent has to be picked out from a realistic mix.
    zero_shot_req = build(data["test_zero_shot"], all_labels, by_ds_all, augment=False)

    # OOS requests: an out-of-scope utterance paired with a plausible small
    # candidate set sampled the same way as any other request (it has no
    # "correct" label, so correct_idx is meaningless here -- only used to
    # check whether the model's confidence in ITS BEST candidate is lower
    # than for genuine in-scope requests, i.e. the same OOS-separation
    # check as earlier iterations, adapted to the per-request candidate
    # framing).
    oos_rng = random.Random(seed + 1)
    oos_req = []
    for text, _ in data["test_oos"]:
        fake_correct = oos_rng.choice(seen_labels)
        req = make_request(text, fake_correct, seen_labels, by_ds_seen, oos_rng, n_options, augment=False)
        oos_req.append(req)

    return {
        "train": train_req, "val": val_req, "test": test_req, "test_zero_shot": zero_shot_req,
        "test_oos": oos_req, "zero_shot_labels": zero_shot_labels,
    }
