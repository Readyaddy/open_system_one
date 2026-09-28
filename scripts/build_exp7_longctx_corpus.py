"""Builds exp7_longctx_reasoning_corpus -- long-context reasoning data with
retained reasoning-depth metadata.

WHY THIS EXISTS
---------------
exp7_hard_reasoning_corpus was built to supply long-context reasoning data and
did not: its longest example is ~1,800 tokens and its median is ~95, because
every source it draws from (ARC, ProofWriter, WMDP) is inherently short. The
"7000" figure it was provisioned against (l_context=7168) is the character
length of its single longest row, not a token count. It also discarded
ProofWriter's QDep field -- the number of inference hops a question actually
requires -- which is the one label that makes "does recurrent depth help?"
directly testable rather than inferable from aggregate accuracy.

WHAT THIS BUILDS
----------------
1. quality           -- QuALITY. Real 4-way MCQ over ~6,100-token articles
                        (max ~8,800). This is the genuine long-context anchor.
                        SPLIT BY ARTICLE: there are only ~150 unique articles
                        with ~17 questions each, so a question-level split
                        would put the same article in train and test.
2. proofwriter_long  -- ProofWriter stratified across QDep (hops required) and
                        padded to a target token length with distractor facts.
                        Gives an explicit (context_length x reasoning_depth)
                        grid: the two variables can be moved independently,
                        which no other source here allows.
3. musr              -- all of it; genuinely long narrative state-tracking.
4. arc_challenge     -- short but hard, kept for task diversity.

Every row keeps: source, qdep (nullable), n_fact, n_rule, hard (nullable),
approx_tokens. The training script can then stratify by reasoning depth
instead of only reporting one aggregate number.

PADDING CORRECTNESS
-------------------
proofwriter_long pads with facts about entities that do NOT appear in the
source theory, using fresh invented names. This is answer-preserving under
ProofWriter's open-world assumption: the question asks about entity E, the
padding only asserts properties of entities E' disjoint from the theory, and
ProofWriter's rules are single-variable universals ("if something is X it is
Y"), so firing them on E' can only derive facts about E' -- never about E.
Padding with *rules* would NOT be safe (a universal rule applies to every
entity including E), and padding with facts that reuse the theory's own names
would not be either. Both are therefore avoided.
"""
import argparse
import hashlib
import json
import os
import random
from collections import Counter, defaultdict

from datasets import load_dataset

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
OUT_DIR = os.path.join(DATA_DIR, "exp7_longctx_reasoning_corpus")

# Fresh entity names for distractor facts. Deliberately disjoint from
# ProofWriter's own cast (Anne/Bob/Charlie/Dave/Erin/Fiona/Gary/Harry and the
# animal nouns), so a padding fact can never collide with a theory entity.
PAD_NAMES = [
    "Ingrid", "Jasper", "Kiran", "Lorna", "Milo", "Nadia", "Osric", "Petra",
    "Quill", "Rosalind", "Soren", "Tamsin", "Ulric", "Vesna", "Wendell",
    "Xiomara", "Yannick", "Zelda", "Ansel", "Briony", "Caspian", "Delphine",
    "Emrys", "Faye", "Gideon", "Hester", "Idris", "Juniper", "Kester", "Linnea",
]
PAD_ADJECTIVES = [
    "big", "blue", "cold", "furry", "green", "kind", "nice", "quiet", "red",
    "rough", "round", "smart", "white", "young",
]


def _write_jsonl(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"    wrote {len(rows):6d} rows -> {path}", flush=True)


def make_pad_sentences(rng, n):
    """Distractor facts about fresh entities. See PADDING CORRECTNESS above."""
    out = []
    for _ in range(n):
        name = rng.choice(PAD_NAMES)
        adj = rng.choice(PAD_ADJECTIVES)
        neg = "not " if rng.random() < 0.3 else ""
        out.append(f"{name} is {neg}{adj}.")
    return out


# --------------------------------------------------------------------------

def build_quality(tok, rng):
    print("  building QuALITY (long-context 4-way MCQ)...", flush=True)
    ds = load_dataset("emozilla/quality")

    # Split by ARTICLE, not by question -- ~17 questions share each article.
    by_article = defaultdict(list)
    for split in ("train", "validation"):
        for ex in ds[split]:
            by_article[ex["article"]].append(ex)

    articles = sorted(by_article.keys())
    rng.shuffle(articles)
    n = len(articles)
    n_tr, n_va = int(n * 0.70), int(n * 0.15)
    assign = {}
    for i, a in enumerate(articles):
        assign[a] = "train" if i < n_tr else ("val" if i < n_tr + n_va else "test")

    rows = {"train": [], "val": [], "test": []}
    for article, exs in by_article.items():
        split = assign[article]
        n_tok = len(tok(article, add_special_tokens=False)["input_ids"])
        for ex in exs:
            options = list(ex["options"])
            ans = int(ex["answer"])
            if not (0 <= ans < len(options)):
                continue
            rows[split].append({
                "context": article,
                "instructions": str(ex["question"]),
                "option_texts": [str(o) for o in options],
                "qtype": "choice",
                "answer_idx": ans,
                "source": "quality",
                "qdep": None,
                "hard": bool(ex.get("hard", False)),
                "approx_tokens": n_tok,
            })
    print(f"    {n} unique articles -> {n_tr} train / {n_va} val / {n - n_tr - n_va} test", flush=True)
    return rows


def _pad_one(tok, rng, theory, pad_pool, mean_pad_len, length_buckets):
    base_tok = len(tok(theory, add_special_tokens=False)["input_ids"])
    target = rng.choice(length_buckets)
    need = max(0, target - base_tok)
    n_pad = int(need / mean_pad_len) if need > 0 else 0
    if n_pad > 0:
        picks = [pad_pool[rng.randrange(len(pad_pool))] for _ in range(n_pad)]
        # Interleave padding around the real theory so the relevant sentences
        # aren't at a fixed position -- otherwise "attend to the first 90
        # tokens" solves every padded example without reading anything.
        cut = rng.randint(0, len(picks))
        context = " ".join(picks[:cut] + [theory] + picks[cut:])
    else:
        context = theory
    return context, target, n_pad, base_tok + int(n_pad * mean_pad_len)


def _negated(question):
    return " not " in f" {question} "


def build_proofwriter_long(tok, rng, per_cell, length_buckets, max_depth=5, balance_polarity=True):
    """ProofWriter, answer-balanced WITHIN each reasoning depth, official splits.

    Two fixes over the first build:

    1. Official HF splits. ProofWriter asks many questions per theory and its
       own train/validation/test are split BY THEORY. The first build drew
       everything from HF train and re-split by question, so the same theory
       (the entire context) appeared in train and val with different questions.

    2. Balanced answers per depth. The natural pool is balanced overall but
       not within depth: Unknown is 57% of depth-0 examples and 16% of
       depth-4+ ones. That makes depth-conditional priors a free shortcut --
       exp7d_best_hard.pt learned exactly that, predicting Unknown on 100% of
       inputs and scoring the Unknown base rate to the decimal. With True /
       False / Unknown equal inside every depth, no prior carries information
       and the only route to accuracy is reading the theory.

    Depths above max_depth are excluded: in the source data depths 6-8 are
    100% Unknown (no True/False exists at those depths), so they would
    reintroduce a pure depth->label leak that cannot be balanced away.

    balance_polarity (v3): in the source pool, provable facts are overwhelmingly
    positive, so True questions are ~92% positively phrased and False questions
    ~92% negated. The word "not" alone therefore decides True vs False, and
    exp7e_v3/v4 learned exactly that rule -- their ProofWriter predictions match
    "'not' -> False, else True, never Unknown" to the decimal in every hop
    bucket, and never changed across three epochs. Balancing each depth over
    (answer x phrasing) makes the phrasing carry zero information about the
    label, so the only route to accuracy is actually evaluating the theory.
    """
    print("  building ProofWriter-Long (official splits, answer-balanced per depth)...", flush=True)
    ds_all = load_dataset("tasksource/proofwriter")
    pad_pool = make_pad_sentences(rng, 4000)
    pad_lens = [len(tok(s, add_special_tokens=False)["input_ids"]) for s in pad_pool]
    mean_pad_len = sum(pad_lens) / len(pad_lens)
    labels = ["True", "False", "Unknown"]

    out = {}
    for split, hf_split in (("train", "train"), ("val", "validation"), ("test", "test")):
        ds = ds_all[hf_split]
        cells = defaultdict(list)
        for i, (d, a, q) in enumerate(zip(ds["QDep"], ds["answer"], ds["question"])):
            try:
                d = int(d)
            except (TypeError, ValueError):
                continue
            if 0 <= d <= max_depth and a in labels:
                pol = _negated(q) if balance_polarity else False
                cells[(d, a, pol)].append(i)

        cap = per_cell[split]
        pols = (False, True) if balance_polarity else (False,)
        chosen = []
        for d in range(max_depth + 1):
            # equal count in every (answer x phrasing) cell of this depth = the rarest cell's count
            n_d = min([cap] + [len(cells[(d, a, pol)]) for a in labels for pol in pols])
            for a in labels:
                for pol in pols:
                    chosen += [(d, i) for i in rng.sample(cells[(d, a, pol)], n_d)]
        rng.shuffle(chosen)

        rows = []
        for d, idx in chosen:
            ex = ds[int(idx)]
            theory = str(ex["theory"]).strip()
            ans = str(ex["answer"]).strip()
            ctx, target, n_pad, approx = _pad_one(tok, rng, theory, pad_pool, mean_pad_len, length_buckets)
            rows.append({
                "context": ctx,
                "instructions": ("Based on the statements provided, is the following true, false, "
                                 f'or unknown: "{str(ex["question"]).strip()}"?'),
                "option_texts": list(labels),
                "qtype": "choice",
                "answer_idx": labels.index(ans),
                "source": "proofwriter_long",
                "qdep": d,
                "negated": _negated(str(ex["question"])),
                "n_fact": int(ex.get("NFact", 0) or 0),
                "n_rule": int(ex.get("NRule", 0) or 0),
                "hard": bool(d >= 2),
                "target_tokens": int(target),
                "n_pad_sentences": n_pad,
                "approx_tokens": approx,
                "theory_hash": hashlib.md5(theory.encode("utf-8")).hexdigest()[:12],
            })
        c = Counter(r["qdep"] for r in rows)
        print(f"    {split}: {len(rows)} rows; rows per depth {dict(sorted(c.items()))} "
              f"(balanced over answer x phrasing={balance_polarity})", flush=True)
        out[split] = rows
    return out


def build_musr(tok, rng):
    print("  building MuSR...", flush=True)
    out = []
    try:
        musr = load_dataset("TAUR-Lab/MuSR", "default")
    except Exception as e:
        print(f"    warning: MuSR failed to load ({type(e).__name__}); skipping", flush=True)
        return {"train": [], "val": [], "test": []}

    for subset in musr.keys():
        for ex in musr[subset]:
            narrative = str(ex.get("narrative", ""))
            question = str(ex.get("question", ""))
            try:
                choices = ex.get("choices")
                if isinstance(choices, str):
                    choices = json.loads(choices.replace("'", '"'))
                choices = [str(c) for c in choices]
            except Exception:
                continue
            try:
                ans = int(ex.get("answer_index", ex.get("answer_choice", -1)))
            except (TypeError, ValueError):
                continue
            if not choices or not (0 <= ans < len(choices)):
                continue
            out.append({
                "context": narrative, "instructions": question,
                "option_texts": choices, "qtype": "choice", "answer_idx": ans,
                "source": f"musr_{subset}", "qdep": None, "hard": True,
                "approx_tokens": len(tok(narrative, add_special_tokens=False)["input_ids"]),
            })
    # Split by NARRATIVE: a narrative can carry several questions, and a
    # question-level split would put the same story in train and test.
    narratives = sorted({r["context"] for r in out})
    rng.shuffle(narratives)
    n = len(narratives)
    n_tr, n_va = int(n * 0.70), int(n * 0.15)
    assign = {t: ("train" if i < n_tr else ("val" if i < n_tr + n_va else "test"))
              for i, t in enumerate(narratives)}
    res = {"train": [], "val": [], "test": []}
    for r in out:
        res[assign[r["context"]]].append(r)
    return res


def build_arc(tok, rng, max_n):
    print("  building ARC-Challenge...", flush=True)
    rows = {"train": [], "val": [], "test": []}
    try:
        ds = load_dataset("allenai/ai2_arc", "ARC-Challenge")
    except Exception as e:
        print(f"    warning: ARC failed to load ({type(e).__name__}); skipping", flush=True)
        return rows
    for split_name, hf in (("train", "train"), ("val", "validation"), ("test", "test")):
        if hf not in ds:
            continue
        for ex in ds[hf]:
            labels, texts = ex["choices"]["label"], ex["choices"]["text"]
            key = ex["answerKey"]
            if key in labels:
                ans = labels.index(key)
            elif key.isdigit() and 0 <= int(key) - 1 < len(texts):
                ans = int(key) - 1
            else:
                continue
            rows[split_name].append({
                "context": "", "instructions": str(ex["question"]),
                "option_texts": [str(t) for t in texts], "qtype": "choice",
                "answer_idx": ans, "source": "arc_challenge", "qdep": None,
                "hard": True, "approx_tokens": 0,
            })
        rows[split_name] = rows[split_name][:max_n]
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pw_train_per_cell", type=int, default=1000,
                     help="ProofWriter examples per (depth, answer) cell in train. The depth-5 "
                          "cell is capped by the rarest class (Unknown has only 447 there).")
    ap.add_argument("--pw_eval_per_cell", type=int, default=120)
    ap.add_argument("--pw_max_depth", type=int, default=5)
    ap.add_argument("--no_polarity_balance", action="store_true",
                     help="Disable answer x phrasing balance (restores the 'not'->False shortcut).")
    ap.add_argument("--length_buckets", type=int, nargs="+",
                     default=[256, 512, 1024, 2048, 4096, 7000],
                     help="Target context token lengths for ProofWriter-Long padding.")
    ap.add_argument("--arc_max", type=int, default=1200)
    ap.add_argument("--no_quality", action="store_true")
    ap.add_argument("--no_musr", action="store_true")
    ap.add_argument("--seed", type=int, default=20260924)
    ap.add_argument("--out_dir", type=str, default=None)
    args = ap.parse_args()

    out_dir = args.out_dir or OUT_DIR
    rng = random.Random(args.seed)

    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments", "exp7_hybrid_decision"))
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-large")

    merged = {"train": [], "val": [], "test": []}

    def add(part):
        for k in merged:
            merged[k].extend(part.get(k, []))

    if not args.no_quality:
        add(build_quality(tok, rng))
    add(build_proofwriter_long(tok, rng,
                                {"train": args.pw_train_per_cell, "val": args.pw_eval_per_cell,
                                 "test": args.pw_eval_per_cell},
                                args.length_buckets, args.pw_max_depth,
                                balance_polarity=not args.no_polarity_balance))
    if not args.no_musr:
        add(build_musr(tok, rng))
    add(build_arc(tok, rng, args.arc_max))

    for k in merged:
        rng.shuffle(merged[k])
        _write_jsonl(merged[k], os.path.join(out_dir, f"{k}.jsonl"))

    print("\n=== corpus summary ===", flush=True)
    for split, rows in merged.items():
        if not rows:
            continue
        toks = sorted(r.get("approx_tokens", 0) for r in rows)
        src = Counter(r["source"] for r in rows)
        print(f"\n{split}: {len(rows)} rows")
        print(f"   approx context tokens: p50={toks[len(toks)//2]} "
              f"p90={toks[int(len(toks)*0.9)]} p99={toks[int(len(toks)*0.99)]} max={toks[-1]}")
        for s, c in src.most_common():
            print(f"     {s:22s} {c:6d} ({c/len(rows)*100:5.1f}%)")
        qd = Counter(r["qdep"] for r in rows if r.get("qdep") is not None)
        if qd:
            print(f"   QDep: { dict(sorted(qd.items())) }")
        for thr in (1024, 2048, 4096, 7000):
            frac = sum(1 for t in toks if t >= thr) / len(toks) * 100
            print(f"     >= {thr:5d} tokens: {frac:5.1f}%")


if __name__ == "__main__":
    main()
