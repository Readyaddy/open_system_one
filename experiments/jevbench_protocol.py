"""Official JevBench item rendering, shared by every evaluator in this repo.

JevBench gives a system the state, the question's instructions AND its
`criteria` -- the sentence that defines each allowed label -- and asks for a
probability per label. Every evaluator here used to pass only the raw label
ids ("pay_subject_to_15000_sublimit", "yes"/"no") and drop the criteria, and
each re-implemented that loading separately, so the mistake was copied four
times. On the public `original` tier the missing criteria alone cost exp7a
23.6 points (73.6% official vs 50.0%).

Keep all JevBench rendering here; evaluators import it rather than parse the
JSONL themselves.

criteria shapes seen in the public tiers:
  choice : {label: definition}
  noul   : {"true": ..., "false": ...} for labels ["no", "yes"]
  score  : [definition per level], in label order
"""
import json
import os

TIERS = ("easy", "original", "hard")
QTYPE = {"choice": "choice", "noul": "bool", "score": "score"}
_BOOL_KEYS = {"yes": "true", "no": "false"}

PUBLIC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "jevbench",
                                          "datasets", "public"))


def option_texts(item):
    """One option string per label, in label order, taken from the criteria.
    Falls back to the bare label only if the item carries no definition for it."""
    labels = [str(l) for l in item["labels"]]
    crit = item.get("question", {}).get("criteria")
    if isinstance(crit, dict):
        out = []
        for l in labels:
            key = l if l in crit else _BOOL_KEYS.get(l.lower(), l)
            out.append(str(crit[key]) if key in crit else l)
        return out
    if isinstance(crit, list) and len(crit) == len(labels):
        return [f"{l}: {c if isinstance(c, str) else json.dumps(c, ensure_ascii=False)}"
                for l, c in zip(labels, crit)]
    return labels


def render(item, tier=""):
    labels = [str(l) for l in item["labels"]]
    state = item.get("state", "")
    return {
        "id": item.get("id"),
        "tier": tier,
        "family": item.get("family") or tier,
        "context": state if isinstance(state, str) else json.dumps(state, ensure_ascii=False),
        "instructions": str(item["question"].get("instructions", "")),
        "option_texts": option_texts(item),
        "labels": labels,
        "qtype": QTYPE.get(item["question"].get("type"), "choice"),
        "answer_idx": labels.index(str(item["expected"])),
    }


def load_tier(tier, public_dir=PUBLIC_DIR):
    path = os.path.join(public_dir, f"{tier}.jsonl")
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [render(json.loads(l), tier) for l in f if l.strip()]
