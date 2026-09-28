"""Experiment 8 data: sampling, example construction, batching, val suite.

Reads the corpora built and audited on 2026-09-24 (scripts/build_exp7_data.py,
filter_exp7_shortcuts.py, weight_exp7_bias.py, audit_exp7_data.py):

  family     source files                          question type
  intent     data/intent_corpus (Banking77 removed) choice, 2-128 options
  mcq        data/exp7_mcq_corpus (AFLite-filtered) choice, 4-5 native (+ padding)
  bool       data/exp7_bool_corpus (bias_w weights) bool
  score      data/exp7_score_corpus                 score (ordinal, RPS)
  diversity  data/exp7_diversity_corpus             choice, 27 label vocabularies

Train-time pattern controls (each one answers an audit finding):
  * class-balanced sampling inside intent/diversity/score sources -- label
    frequency never becomes a prior (intent labels range 4..1803 examples)
  * bool rows drawn proportionally to bias_w -- down-weights examples a
    blind (question-only / context-only) model gets right
  * source weight propto sqrt(size), so small sources are seen but don't dominate
  * option order shuffled for every non-ordinal question, including yes/no
  * diversity: a random SUBSET of the native label set half the time, so the
    answer isn't always "the one coherent vocabulary among foreign padding"
  * score: stored option texts (descriptive levels, 30% listed high->low),
    never reordered -- RPS needs ordinal order, and direction must be read
"""
import json
import math
import os
import random
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from dataset_v7 import _humanize, raw_intent_name, sample_outcome_description  # noqa: E402

from s1_model import PackedExample, PackedBatch, QTYPE_IDX  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import jevbench_protocol as JB  # noqa: E402

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data"))

INSTR_CHOICE = [
    "Which category best describes what this is about, or where should it be routed?",
    "Pick the option that best matches this request.",
    "Which of the following applies here?",
    "Choose the single best-fitting category.",
    "Where should this be routed?",
    "",
]
BOOL_OPTION_WORDS = [("no", "yes"), ("false", "true"), ("No", "Yes")]


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


# --------------------------------------------------------------------------
# Example metadata travels with every PackedExample (for per-group metrics)
# --------------------------------------------------------------------------

@dataclass
class Meta:
    family: str
    source: str
    qtype: str
    n_options: int
    blind_easy: bool = False
    group: str = ""          # val group name
    extra: dict = field(default_factory=dict)


def sample_modality(rng, p_vec_only, p_text_only):
    r = rng.random()
    if r < p_vec_only:
        return False, True
    if r < p_vec_only + p_text_only:
        return True, False
    return True, True


def sample_n(rng, lo=2, hi=128, p_max=0.03, max_n=255):
    if rng.random() < p_max:
        return max_n
    return int(round(math.exp(rng.uniform(math.log(lo), math.log(hi)))))


# --------------------------------------------------------------------------
# Builders
# --------------------------------------------------------------------------

def _tokset(name):
    return set(_humanize(name).lower().split())


def intent_distractors(true_label, pool, k, rng, thr=0.5):
    tt = _tokset(raw_intent_name(true_label))
    scored = []
    for c in pool:
        if c == true_label:
            continue
        ct = _tokset(raw_intent_name(c))
        u = tt | ct
        scored.append((len(tt & ct) / len(u) if u else 0.0, c))
    safe = [c for j, c in scored if j < thr]
    if len(safe) >= k:
        return rng.sample(safe, k)
    scored.sort(key=lambda x: x[0])
    return [c for _, c in scored[:k]]


def render_intent(label, mode, rng):
    raw = raw_intent_name(label)
    phrase = _humanize(raw)
    if mode == "bare":
        return phrase
    if mode == "template":
        return sample_outcome_description(raw, rng)
    if mode == "description":
        return f"The user wants help with: {phrase}."
    return f"{phrase} -- {sample_outcome_description(raw, rng)}"


def build_intent(text, label, label_pool, rng, cfg, n=None, instructions=None, family="intent", source="intent"):
    n = min(n or sample_n(rng), len(label_pool))
    opts = intent_distractors(label, label_pool, n - 1, rng) + [label]
    rng.shuffle(opts)
    mode = rng.choice(["bare", "template", "description", "label_description"])
    ut, uv = sample_modality(rng, cfg.p_vec_only, cfg.p_text_only)
    instr = rng.choice(INSTR_CHOICE) if instructions is None else instructions
    ex = PackedExample(context=text, instructions=instr, option_texts=[render_intent(l, mode, rng) for l in opts],
                       qtype="choice", answer_idx=opts.index(label), use_text=ut, use_vector=uv, source=source)
    return ex, Meta(family, source, "choice", len(opts))


def build_mcq(row, foreign, rng, cfg):
    opts = list(row["options"])
    true = opts[row["answer_idx"]]
    n_t = min(sample_n(rng, hi=20), len(opts) + len(foreign))
    if n_t > len(opts) and foreign and rng.random() < 0.5:
        opts += rng.sample(foreign, n_t - len(opts))
    rng.shuffle(opts)
    ut, uv = sample_modality(rng, cfg.p_vec_only, cfg.p_text_only)
    instr = f"{rng.choice(INSTR_CHOICE[1:4])} {row['question']}".strip()
    ex = PackedExample(context=row.get("context") or "", instructions=instr, option_texts=opts, qtype="choice",
                       answer_idx=opts.index(true), use_text=ut, use_vector=uv, source=f"mcq:{row['source']}")
    return ex, Meta("mcq", row["source"], "choice", len(opts), bool(row.get("blind_easy")))


# Descriptive-criteria renderings of the two bool outcomes, the format Jev's
# `noul` questions use (typed-decisions: "No human attention is warranted." vs
# "A human should inspect this run."). exp8a3 only ever saw yes/no words, and
# on typed-decisions bool it predicted the second option for every item (57.2%
# = exactly the rate at which option 2 is correct). Each pair is (yes-outcome,
# no-outcome). Negation appears on BOTH sides across the bank, so "the option
# containing 'not'" carries no label information.
BOOL_CRITERIA = {
    "boolq": [
        ("The passage indicates the answer is yes.", "The passage indicates the answer is no."),
        ("Yes -- the text confirms this.", "No -- the text shows this is not the case."),
        ("This is true according to the passage.", "This is false according to the passage."),
        ("The statement in the question holds.", "The statement in the question does not hold."),
    ],
    "mnli": [
        ("The text supports this claim.", "The text does not establish this claim."),
        ("The claim follows from what is written.", "The claim is contradicted or left unsupported by the text."),
        ("Reading the text, the claim must be true.", "Reading the text, the claim could be false."),
        ("Nothing is missing: the text entails the claim.", "Something is missing or conflicting: the text does not entail the claim."),
    ],
    "paws": [
        ("Both sentences say the same thing.", "The sentences differ in meaning."),
        ("These are paraphrases of each other.", "These are not paraphrases; the meaning changes."),
        ("No meaning is lost or changed between them.", "Some meaning is lost or changed between them."),
    ],
    "qqp": [
        ("Both questions ask for the same thing.", "The questions ask for different things."),
        ("These are duplicate questions.", "These are not duplicates."),
        ("One answer would satisfy both questions.", "Each question needs its own answer."),
    ],
}


def build_bool(row, rng, cfg, fixed_words=None, descriptive=None):
    """descriptive: None -> sample with cfg.p_bool_desc; True/False forces it.
    Option order is shuffled for both renderings (fixed_words pins yes/no
    wording AND the no-first order, for the exp7a-comparable val groups)."""
    if descriptive is None:
        descriptive = fixed_words is None and rng.random() < getattr(cfg, "p_bool_desc", 0.0)
    if descriptive:
        yes, no = rng.choice(BOOL_CRITERIA.get(row["source"], BOOL_CRITERIA["boolq"]))
        yes_first = rng.random() < 0.5
    else:
        no, yes = fixed_words or rng.choice(BOOL_OPTION_WORDS)
        yes_first = rng.random() < 0.5 if fixed_words is None else False
    opts = [yes, no] if yes_first else [no, yes]
    ans = opts.index(yes if row["answer_idx"] == 1 else no)
    ut, uv = sample_modality(rng, cfg.p_vec_only, cfg.p_text_only)
    ex = PackedExample(context=row["context"], instructions=row["question"], option_texts=opts, qtype="bool",
                       answer_idx=ans, use_text=ut, use_vector=uv, source=f"bool:{row['source']}")
    return ex, Meta("bool", row["source"], "bool", 2, bool(row.get("blind_easy")),
                    extra={"descriptive": descriptive, "yes_first": yes_first})


def build_score(row, rng, cfg):
    ut, uv = sample_modality(rng, cfg.p_vec_only, cfg.p_text_only)
    ex = PackedExample(context=row["context"], instructions=row["question"], option_texts=list(row["options"]),
                       qtype="score", answer_idx=int(row["answer_idx"]), use_text=ut, use_vector=uv,
                       source=f"score:{row['source']}")
    return ex, Meta("score", row["source"], "score", len(row["options"]), bool(row.get("blind_easy")),
                    extra={"reversed": row.get("reversed", False)})


class DiversityBuilder:
    def __init__(self, rows, instructions):
        self.instructions = instructions
        by_src = defaultdict(set)
        for r in rows:
            by_src[r["source"]].update(r["labels"])
        self.foreign = {s: [l for t, ls in by_src.items() if t != s for l in ls] for s in by_src}

    def instr(self, row, rng):
        src = row["source"]
        key = "tweet_stance" if src.startswith("tweet_stance") else src
        bank = self.instructions.get(key) or INSTR_CHOICE
        return rng.choice(bank).replace("{target}", row.get("target", "this topic"))

    def __call__(self, row, rng, cfg, full=False):
        native = list(row["labels"])
        true = row["label"]
        if full:
            opts = native
        else:
            if len(native) > 2 and rng.random() < 0.5:
                k = rng.randint(2, len(native))
                opts = [true] + rng.sample([l for l in native if l != true], k - 1)
            else:
                opts = native
            n_t = sample_n(rng, hi=64)
            if n_t > len(opts) and rng.random() < 0.3:
                pool = [l for l in self.foreign[row["source"]] if l not in opts]
                opts = opts + rng.sample(pool, min(n_t - len(opts), len(pool)))
        opts = list(opts)
        rng.shuffle(opts)
        ut, uv = sample_modality(rng, cfg.p_vec_only, cfg.p_text_only) if not full else (True, True)
        ex = PackedExample(context=row["text"], instructions=self.instr(row, rng), option_texts=opts,
                           qtype="choice", answer_idx=opts.index(true), use_text=ut, use_vector=uv,
                           source=f"diversity:{row['source']}")
        return ex, Meta("diversity", row["source"], "choice", len(opts), bool(row.get("blind_easy")))


# --------------------------------------------------------------------------
# Corpora + training sampler
# --------------------------------------------------------------------------

@dataclass
class DataCfg:
    w_intent: float = 0.20
    w_mcq: float = 0.25
    w_bool: float = 0.15
    w_score: float = 0.15
    w_diversity: float = 0.25
    p_vec_only: float = 0.10
    p_text_only: float = 0.10
    hellaswag_mult: float = 0.5
    qqp_mult: float = 0.5
    p_bool_desc: float = 0.0   # exp8b: fraction of bool examples rendered as descriptive criteria


def load_corpora():
    C = {}
    intent = {s: read_jsonl(os.path.join(DATA_DIR, "intent_corpus", f"{s}.jsonl"))
              for s in ("train", "val", "test_zero_shot")}
    labels = json.load(open(os.path.join(DATA_DIR, "intent_corpus", "labels.json"), encoding="utf-8"))
    all_rows = intent["train"] + intent["val"] + read_jsonl(os.path.join(DATA_DIR, "intent_corpus", "test.jsonl"))
    C["intent"] = {
        "train": [r for r in intent["train"] if not r["label"].startswith("banking::")],
        "val": [r for r in intent["val"] if not r["label"].startswith("banking::")],
        "zero_shot": intent["test_zero_shot"],
        "seen_labels": [l for l in labels["seen_labels"] if not l.startswith("banking::")],
        "all_labels": [l for l in labels["all_labels"] if not l.startswith("banking::")],
        "banking77": [r for r in all_rows if r["label"].startswith("banking::")],
    }
    C["banking77_labels"] = sorted({r["label"] for r in C["intent"]["banking77"]})
    for fam, d in (("mcq", "exp7_mcq_corpus"), ("bool", "exp7_bool_corpus"),
                   ("score", "exp7_score_corpus"), ("diversity", "exp7_diversity_corpus")):
        C[fam] = {s: read_jsonl(os.path.join(DATA_DIR, d, f"{s}.jsonl")) for s in ("train", "val")}
    C["diversity_instructions"] = json.load(open(os.path.join(DATA_DIR, "exp7_diversity_corpus", "instructions.json"),
                                                 encoding="utf-8"))
    return C


class TrainSampler:
    """Three-level sampling: family (fixed weights) -> source (sqrt size, with
    per-source multipliers) -> row (class-balanced, or bias_w for bool)."""

    def __init__(self, C, cfg: DataCfg, seed=0):
        self.cfg = cfg
        self.rng = random.Random(seed)
        self.div = DiversityBuilder(C["diversity"]["train"], C["diversity_instructions"])
        self.intent_labels = C["intent"]["seen_labels"]
        self.mcq_foreign = [o for r in C["mcq"]["train"][:20000] for o in r["options"]]

        self.fam_names = ["intent", "mcq", "bool", "score", "diversity"]
        self.fam_w = [cfg.w_intent, cfg.w_mcq, cfg.w_bool, cfg.w_score, cfg.w_diversity]
        self.src = {}
        # intent: class-balanced over labels
        by_lab = defaultdict(list)
        for r in C["intent"]["train"]:
            by_lab[r["label"]].append(r)
        self.src["intent"] = ([("intent", {"by_label": by_lab, "labels": list(by_lab)})], [1.0])
        for fam in ("mcq", "bool", "score", "diversity"):
            by_s = defaultdict(list)
            for r in C[fam]["train"]:
                by_s[r["source"]].append(r)
            entries, ws = [], []
            for s, rows in sorted(by_s.items()):
                mult = cfg.hellaswag_mult if s == "hellaswag" else (cfg.qqp_mult if s == "qqp" else 1.0)
                if fam == "bool":
                    pool = {"rows": rows, "cum": list(_cumsum([r.get("bias_w", 1.0) for r in rows]))}
                elif fam in ("score", "diversity"):
                    key = "level" if fam == "score" else "label"
                    bl = defaultdict(list)
                    for r in rows:
                        bl[r[key]].append(r)
                    pool = {"by_label": bl, "labels": list(bl)}
                else:
                    pool = {"rows": rows}
                entries.append((s, pool))
                ws.append(math.sqrt(len(rows)) * mult)
            self.src[fam] = (entries, ws)

    def mix_table(self):
        out = {}
        tot_f = sum(self.fam_w)
        for f, wf in zip(self.fam_names, self.fam_w):
            entries, ws = self.src[f]
            for (s, _), w in zip(entries, ws):
                out[f"{f}/{s}"] = wf / tot_f * w / sum(ws)
        return out

    def sample(self):
        rng = self.rng
        fam = rng.choices(self.fam_names, weights=self.fam_w)[0]
        entries, ws = self.src[fam]
        s, pool = entries[rng.choices(range(len(entries)), weights=ws)[0]]
        if "by_label" in pool:
            row = rng.choice(pool["by_label"][rng.choice(pool["labels"])])
        elif "cum" in pool:
            row = pool["rows"][_bisect(pool["cum"], rng.random() * pool["cum"][-1])]
        else:
            row = rng.choice(pool["rows"])
        cfg = self.cfg
        if fam == "intent":
            return build_intent(row["text"], row["label"], self.intent_labels, rng, cfg)
        if fam == "mcq":
            return build_mcq(row, self.mcq_foreign, rng, cfg)
        if fam == "bool":
            return build_bool(row, rng, cfg)
        if fam == "score":
            return build_score(row, rng, cfg)
        return self.div(row, rng, cfg)


def _cumsum(xs):
    t = 0.0
    for x in xs:
        t += x
        yield t


def _bisect(cum, x):
    lo, hi = 0, len(cum) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if cum[mid] < x:
            lo = mid + 1
        else:
            hi = mid
    return lo


# --------------------------------------------------------------------------
# Batching: tokenize once, pack into micro-batches under a token budget
# --------------------------------------------------------------------------

class MicroBatch:
    """CPU tensors for one forward pass; .to(device) moves them (non-blocking
    when pinned)."""

    def __init__(self, packed: PackedBatch, opt: dict, metas: List[Meta]):
        self.packed, self.opt, self.metas = packed, opt, metas

    def pin_memory(self):
        p = self.packed
        for f in ("input_ids", "attention_mask", "context_token_mask", "mask_positions", "valid_mask",
                  "inject_scale", "text_scale", "qtype_idx", "answer_idx"):
            setattr(p, f, getattr(p, f).pin_memory())
        self.opt = {k: v.pin_memory() for k, v in self.opt.items()}
        return self

    def to(self, device):
        p = self.packed
        nb = dict(non_blocking=True)
        packed = PackedBatch(
            input_ids=p.input_ids.to(device, **nb), attention_mask=p.attention_mask.to(device, **nb),
            context_token_mask=p.context_token_mask.to(device, **nb), mask_positions=p.mask_positions.to(device, **nb),
            valid_mask=p.valid_mask.to(device, **nb), inject_scale=p.inject_scale.to(device, **nb),
            text_scale=p.text_scale.to(device, **nb), qtype_idx=p.qtype_idx.to(device, **nb),
            answer_idx=p.answer_idx.to(device, **nb), option_full_texts=None)
        return packed, {k: v.to(device, **nb) for k, v in self.opt.items()}

    @property
    def n_tokens(self):
        return int(self.packed.attention_mask.sum())


class Packer:
    def __init__(self, builder, tok, max_tokens=12000, max_options=512, opt_max_len=64, pad_multiple=64):
        self.builder, self.tok = builder, tok
        self.max_tokens, self.max_options = max_tokens, max_options
        self.opt_max_len, self.pad_multiple = opt_max_len, pad_multiple

    def _assemble(self, items):
        """items: list of (PackedExample, Meta, (ids, mask_pos, ctx_mask))."""
        L = max(len(b[0]) for _, _, b in items)
        L = -(-L // self.pad_multiple) * self.pad_multiple
        N = max(len(b[1]) for _, _, b in items)
        B = len(items)
        pad = self.tok.pad_token_id
        ids = torch.full((B, L), pad, dtype=torch.long)
        am = torch.zeros((B, L), dtype=torch.long)
        cm = torch.zeros((B, L), dtype=torch.bool)
        mp = torch.full((B, N), -1, dtype=torch.long)
        vm = torch.zeros((B, N), dtype=torch.bool)
        for i, (_, _, (x, m, c)) in enumerate(items):
            ids[i, :len(x)] = torch.tensor(x)
            am[i, :len(x)] = 1
            cm[i, :len(x)] = torch.tensor(c)
            mp[i, :len(m)] = torch.tensor(m)
            vm[i, :len(m)] = True
        exs = [e for e, _, _ in items]
        packed = PackedBatch(
            input_ids=ids, attention_mask=am, context_token_mask=cm, mask_positions=mp, valid_mask=vm,
            inject_scale=torch.tensor([1.0 if e.use_vector else 0.0 for e in exs]),
            text_scale=torch.tensor([1.0 if e.use_text else 0.0 for e in exs]),
            qtype_idx=torch.tensor([QTYPE_IDX[e.qtype] for e in exs]),
            answer_idx=torch.tensor([e.answer_idx for e in exs]), option_full_texts=None)
        uniq, index, owner, slot = {}, [], [], []
        for b, e in enumerate(exs):
            for j, t in enumerate(e.option_texts):
                index.append(uniq.setdefault(t, len(uniq)))
                owner.append(b)
                slot.append(j)
        enc = self.tok(list(uniq), padding=True, truncation=True, max_length=self.opt_max_len, return_tensors="pt")
        opt = {"ids": enc["input_ids"], "mask": enc["attention_mask"], "index": torch.tensor(index),
               "owner": torch.tensor(owner), "slot": torch.tensor(slot)}
        return MicroBatch(packed, opt, [m for _, m, _ in items])

    def pack(self, pairs):
        """pairs: list of (PackedExample, Meta) -> list of MicroBatch, length-sorted,
        each under max_tokens (padded length x rows) and max_options."""
        items = [(e, m, self.builder.build_one(e)) for e, m in pairs]
        items.sort(key=lambda it: len(it[2][0]))
        out, cur = [], []
        for it in items:
            L = -(-len(it[2][0]) // self.pad_multiple) * self.pad_multiple
            n_opt = sum(len(c[0].option_texts) for c in cur) + len(it[0].option_texts)
            if cur and (L * (len(cur) + 1) > self.max_tokens or n_opt > self.max_options):
                out.append(self._assemble(cur))
                cur = []
            cur.append(it)
        if cur:
            out.append(self._assemble(cur))
        return out


class StepStream(torch.utils.data.IterableDataset):
    """Each item = one optimizer step's worth of micro-batches (examples_per_step
    examples). Workers get distinct seeds, so steps never repeat across workers."""

    def __init__(self, C, cfg, builder, tok, examples_per_step, max_tokens, seed, max_options=256):
        self.C, self.cfg, self.builder, self.tok = C, cfg, builder, tok
        self.eps, self.max_tokens, self.seed = examples_per_step, max_tokens, seed
        self.max_options = max_options

    def __iter__(self):
        wi = torch.utils.data.get_worker_info()
        wid = wi.id if wi else 0
        sampler = TrainSampler(self.C, self.cfg, seed=self.seed * 1000 + wid)
        packer = Packer(self.builder, self.tok, max_tokens=self.max_tokens, max_options=self.max_options)
        while True:
            yield packer.pack([sampler.sample() for _ in range(self.eps)])


def collate_identity(x):
    return x[0]


# --------------------------------------------------------------------------
# Validation suite -- fixed, deterministic, every group tagged
# --------------------------------------------------------------------------

AG_NEWS_OPTIONS = ["World news", "Sports news", "Business news", "Science and technology news"]
EMOTION_OPTIONS = ["sadness", "joy", "love", "anger", "fear", "surprise"]


def build_val_suite(C, cfg: DataCfg, sizes=None, seed=20260924, external=True, with_bool_desc=False):
    """Returns {group_name: [(PackedExample, Meta), ...]}. Groups:
      intent/seen@50, intent/zero_shot@50           (exp7a-comparable, N=50)
      banking77/generic, banking77/routing          (TRUE zero-shot, all 71 labels)
      mcq/<src>, bool/<src>, score/<src>, diversity/<src>   (per source, blind_easy tagged)
      typed/<qtype>                                 (target-domain eval, never trained on)
      external/ag_news, external/dair_emotion       (Laya/Jev comparison, never trained on)
      jevbench_<tier>/<qtype>                       (report-only; public easy 48 / original 72 / hard 111,
                                                     official protocol: criteria as options, long context)
    Every example uses both modalities (no dropout) -- that's the deployed setting."""
    sizes = sizes or {}
    n_int = sizes.get("intent", 1000)
    n_src = sizes.get("per_source", 400)
    rng = random.Random(seed)
    desc_rng = random.Random(seed + 17)
    full = DataCfg(**{**cfg.__dict__, "p_vec_only": 0.0, "p_text_only": 0.0})
    G = {}

    def take(rows, n):
        rows = list(rows)
        random.Random(seed).shuffle(rows)
        return rows[:n]

    it = C["intent"]
    G["intent/seen@50"] = [build_intent(r["text"], r["label"], it["seen_labels"], rng, full, n=50)
                           for r in take(it["val"], n_int)]
    G["intent/zero_shot@50"] = [build_intent(r["text"], r["label"], it["all_labels"], rng, full, n=50,
                                             source="zero_shot") for r in take(it["zero_shot"], n_int)]
    b77 = take(it["banking77"], n_int)
    nb = len(C["banking77_labels"])
    G["banking77/generic"] = [build_intent(r["text"], r["label"], C["banking77_labels"], rng, full, n=nb,
                                           family="banking77", source="generic") for r in b77]
    G["banking77/routing"] = [build_intent(r["text"], r["label"], C["banking77_labels"], rng, full, n=nb,
                                           instructions="Which category should this be routed to?",
                                           family="banking77", source="routing") for r in b77]

    for fam in ("mcq", "bool", "score", "diversity"):
        by = defaultdict(list)
        for r in C[fam]["val"]:
            by[r["source"]].append(r)
        div = DiversityBuilder(C["diversity"]["train"], C["diversity_instructions"]) if fam == "diversity" else None
        for s, rows in sorted(by.items()):
            rows = take(rows, n_src)
            if fam == "mcq":
                G[f"mcq/{s}"] = [build_mcq(r, [], rng, full) for r in rows]
            elif fam == "bool":
                G[f"bool/{s}"] = [build_bool(r, rng, full, fixed_words=("no", "yes")) for r in rows]
                if with_bool_desc:
                    # Same rows, descriptive criteria, order shuffled -- the Jev `noul` format.
                    # Own RNG, so every pre-existing val group stays identical to exp8a3's.
                    G[f"bool/{s}@desc"] = [build_bool(r, desc_rng, full, descriptive=True) for r in rows]
            elif fam == "score":
                G[f"score/{s}"] = [build_score(r, rng, full) for r in rows]
            else:
                G[f"diversity/{s}"] = [div(r, rng, full, full=True) for r in rows]

    td_path = os.path.join(DATA_DIR, "exp7_typed_decisions", "test.jsonl")
    if os.path.exists(td_path):
        by = defaultdict(list)
        for r in read_jsonl(td_path):
            by[r["qtype"]].append(r)
        for q, rows in by.items():
            G[f"typed/{q}"] = [(PackedExample(context=r["context"], instructions=r["question"],
                                              option_texts=r["options"], qtype=q, answer_idx=r["answer_idx"]),
                                Meta("typed", r.get("workflow", ""), q, len(r["options"])))
                               for r in take(rows, sizes.get("typed", 500))]
            if with_bool_desc and q == "bool":
                # Same items, the two options swapped. A position-driven model's
                # accuracy flips to (1 - acc); a content-driven one's holds.
                G["typed/bool@swapped"] = [(PackedExample(context=r["context"], instructions=r["question"],
                                                          option_texts=r["options"][::-1], qtype=q,
                                                          answer_idx=1 - r["answer_idx"]),
                                            Meta("typed", r.get("workflow", ""), q, 2))
                                           for r in take(rows, sizes.get("typed", 500))]

    # JevBench public tiers, rendered by the official protocol (criteria as the
    # option text; full state). The old "hard111/<q>" groups passed bare label
    # ids and are replaced -- numbers under the new names are NOT comparable to
    # the old hard111 curves. run_suite evaluates these at long context.
    for tier in JB.TIERS:
        by = defaultdict(list)
        for x in JB.load_tier(tier):
            by[x["qtype"]].append((PackedExample(context=x["context"], instructions=x["instructions"],
                                                 option_texts=x["option_texts"], qtype=x["qtype"],
                                                 answer_idx=x["answer_idx"]),
                                   Meta(f"jevbench_{tier}", x["family"], x["qtype"], len(x["labels"]))))
        for q, xs in by.items():
            G[f"jevbench_{tier}/{q}"] = xs

    if external:
        try:
            from datasets import load_dataset
            ag = load_dataset("fancyzhx/ag_news", split="test").select(range(1000))
            G["external/ag_news"] = [(PackedExample(context=e["text"], instructions="Which category best describes this news article?",
                                                    option_texts=AG_NEWS_OPTIONS, qtype="choice", answer_idx=e["label"]),
                                      Meta("external", "ag_news", "choice", 4)) for e in ag]
            em = load_dataset("dair-ai/emotion", split="test").select(range(1000))
            G["external/dair_emotion"] = [(PackedExample(context=e["text"], instructions="What emotion does this text express?",
                                                         option_texts=EMOTION_OPTIONS, qtype="choice", answer_idx=e["label"]),
                                           Meta("external", "dair_emotion", "choice", 6)) for e in em]
        except Exception as e:  # offline etc.
            print(f"  external benchmarks skipped: {e}", flush=True)

    for g, xs in G.items():
        for _, m in xs:
            m.group = g
    return G


# Which val groups make up the model-selection score. External benchmarks,
# typed-decisions and Hard-111 are deliberately EXCLUDED so they stay honest
# held-out numbers rather than something checkpoints are chosen on.
def selection_groups(groups):
    return [g for g in groups if g.split("/")[0] in ("intent", "mcq", "bool", "score", "diversity")]
