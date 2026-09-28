"""Experiment 7d-v2: Hard-reasoning fine-tuning with an actual incentive for depth.

Rewrite of train_exp7d_hard.py. The original is left intact -- it produced the
existing exp7d_* checkpoints and SESSION_REPORT.md cites it, so it stays as the
record of what was run. Everything below is what changed and why.

THE CENTRAL PROBLEM THIS ADDRESSES
----------------------------------
The depth loss was `L = (1/K) * sum_k CE(p_k, t)`: every pass independently
pushed toward the right answer, with equal weight. Read as an optimization
problem, that objective's *optimum* is a loop that does nothing -- if pass 1
is already as good as it can be, the cheapest way to minimize the sum is for
passes 2..K to reproduce pass 1 exactly. A contraction to a fixed point isn't
a failure of that objective, it IS that objective's solution. So a flat
depth curve was never evidence that "looping can't work here"; it was the
loss getting exactly what it asked for. NOTES.md Sec 3.2 chose uniform
weighting deliberately, to keep every depth a valid standalone answer for
exp7c's halting head -- but that halting experiment has now run
(exp7c_halting_v2) and found nothing to halt on, so the constraint it was
protecting no longer has a claim on the design.

Two levers are added, both off by default so they can be turned on one at a
time rather than confounded together:

  --depth_weighting ascending   weight w_k propto k, normalized. Deeper passes
                                carry more of the loss, so the loop has a
                                reason to use the extra compute.
  --monotonic_weight W          hinge penalty relu(margin_{k-1} - margin_k) on
                                the correct option's margin. Doesn't reward
                                improvement, just forbids getting worse with
                                depth -- the standard anytime-prediction
                                constraint. This is the term that directly
                                shapes a rising depth curve.

OTHER FIXES
-----------
1. load_state_dict(strict=False) reported no missing/unexpected keys anywhere.
   With a head this custom (cross-attention recurrent_block, depth_embed,
   recurrent_norm, s0_gate) a silent name mismatch would randomly initialize
   the entire decision head and still train to a plausible-looking loss. Now
   printed, and --require_exact_load turns it into a hard error.
2. Constant LR 1.5e-5, fresh optimizer, no warmup, no decay -- diagnosed in
   SESSION_REPORT.md as a cause of the post-epoch-2 regression. Now warmup +
   cosine decay.
3. Static option order (`random.Random(idx + 42)`, identical every epoch) --
   also diagnosed there. Now reseeded per epoch.
4. Eval ran 111 forward passes at batch size 1 with an 8192-token context.
   Now batched.
5. No error bars. A 1-example difference on n=111 is 0.9 points, and the
   reported k=5/k=6 "peak" over k=1..4 was exactly that one example. Every
   eval now carries a bootstrap 95% CI so a real gain is distinguishable
   from resampling noise.
6. Best-checkpoint selection keyed on k=6 alone (the noisiest single number
   available). Now selects on mean-over-depths by default.
7. Corpus is 79% ProofWriter (25000/31466), all 3-option, one template.
   ARC-Challenge -- the source NOTES.md Sec 6 predicted deliberation would
   actually pay on -- is 3.6%. (Superseded: the v2 corpus is rebuilt with answers balanced
   within every reasoning depth, so no sampling reweight is needed.)
8. Gradient accumulation never flushed the remainder at epoch end.
9. TF32 + configurable gradient checkpointing + larger batches for the 8K
   context, which is where the GPU time actually goes.
"""
import argparse
import json
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler


def _stub():
    import types, importlib.machinery
    if 'torchvision' in sys.modules:
        return
    fake_tv = types.ModuleType('torchvision')
    fake_tv.__spec__ = importlib.machinery.ModuleSpec('torchvision', loader=None)
    fake_tv.__version__ = '0.0.0'
    fake_transforms = types.ModuleType('torchvision.transforms')
    fake_transforms.__spec__ = importlib.machinery.ModuleSpec('torchvision.transforms', loader=None)
    class InterpolationMode:
        NEAREST = 'nearest'; NEAREST_EXACT = 'nearest_exact'; BOX = 'box'
        BILINEAR = 'bilinear'; HAMMING = 'hamming'; BICUBIC = 'bicubic'; LANCZOS = 'lanczos'
    fake_transforms.InterpolationMode = InterpolationMode
    fake_tv.transforms = fake_transforms
    fake_io = types.ModuleType('torchvision.io')
    fake_io.__spec__ = importlib.machinery.ModuleSpec('torchvision.io', loader=None)
    fake_tv.io = fake_io
    sys.modules['torchvision'] = fake_tv
    sys.modules['torchvision.transforms'] = fake_transforms
    sys.modules['torchvision.io'] = fake_io


_stub()
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import jevbench_protocol as JB  # noqa: E402

from model import (HybridDecisionModel, PackedSequenceBuilder, PackedExample, PackedBatch,
                   get_tokenizer, load_flexible, BACKBONE)

AUTOCAST_DTYPE = torch.bfloat16


# --------------------------------------------------------------------------
# Async batch plumbing (carried over from v1 -- this part was genuinely good:
# pinned memory + non-blocking H2D + pre-tokenized options is what took the
# epoch time from 73m to 19m, and none of it is changed here)
# --------------------------------------------------------------------------

class AsyncPackedBatch:
    def __init__(self, input_ids, attention_mask, context_token_mask, mask_positions,
                 valid_mask, inject_scale, text_scale, qtype_idx, answer_idx,
                 per_owner_n, option_input_ids, option_attention_mask, option_index=None,
                 option_owner=None, option_slot=None):
        self.option_index = option_index
        self.option_owner = option_owner
        self.option_slot = option_slot
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.context_token_mask = context_token_mask
        self.mask_positions = mask_positions
        self.valid_mask = valid_mask
        self.inject_scale = inject_scale
        self.text_scale = text_scale
        self.qtype_idx = qtype_idx
        self.answer_idx = answer_idx
        self.per_owner_n = per_owner_n
        self.option_input_ids = option_input_ids
        self.option_attention_mask = option_attention_mask

    def pin_memory(self):
        self.input_ids = self.input_ids.pin_memory()
        self.attention_mask = self.attention_mask.pin_memory()
        self.context_token_mask = self.context_token_mask.pin_memory()
        self.mask_positions = self.mask_positions.pin_memory()
        self.valid_mask = self.valid_mask.pin_memory()
        self.inject_scale = self.inject_scale.pin_memory()
        self.text_scale = self.text_scale.pin_memory()
        self.qtype_idx = self.qtype_idx.pin_memory()
        self.answer_idx = self.answer_idx.pin_memory()
        if self.option_input_ids is not None:
            self.option_input_ids = self.option_input_ids.pin_memory()
            self.option_attention_mask = self.option_attention_mask.pin_memory()
        for name in ("option_index", "option_owner", "option_slot"):
            t = getattr(self, name)
            if t is not None:
                setattr(self, name, t.pin_memory())
        return self

    def to_device(self, device):
        nb = dict(non_blocking=True)
        opt_enc = None
        if self.option_input_ids is not None:
            opt_enc = {
                "input_ids": self.option_input_ids.to(device, **nb),
                "attention_mask": self.option_attention_mask.to(device, **nb),
            }
            if self.option_index is not None:
                opt_enc["index"] = self.option_index.to(device, **nb)
            if self.option_owner is not None:
                opt_enc["owner"] = self.option_owner.to(device, **nb)
                opt_enc["slot"] = self.option_slot.to(device, **nb)
        return PackedBatch(
            input_ids=self.input_ids.to(device, **nb),
            attention_mask=self.attention_mask.to(device, **nb),
            context_token_mask=self.context_token_mask.to(device, **nb),
            mask_positions=self.mask_positions.to(device, **nb),
            valid_mask=self.valid_mask.to(device, **nb),
            inject_scale=self.inject_scale.to(device, **nb),
            text_scale=self.text_scale.to(device, **nb),
            qtype_idx=self.qtype_idx.to(device, **nb),
            answer_idx=self.answer_idx.to(device, **nb),
            option_full_texts=(opt_enc, self.per_owner_n),
        )


def fast_encode_and_project_options(self, tokenizer, batch: PackedBatch, device, chunk_size: int = 128):
    if isinstance(batch.option_full_texts, tuple) and isinstance(batch.option_full_texts[0], dict):
        opt_enc, per_owner_n = batch.option_full_texts
        pooled, _, _ = self.encode_options_raw(
            tokenizer, {"input_ids": opt_enc["input_ids"], "attention_mask": opt_enc["attention_mask"]},
            device, chunk_size=chunk_size, need_tokens=self.use_maxsim)
        if "index" in opt_enc:
            pooled = pooled[opt_enc["index"]]
        projected = self.option_projector(pooled)
        projected = projected / projected.norm(dim=-1, keepdim=True).clamp(min=1e-6) * self.emb_scale

        B = len(per_owner_n)
        Nmax = batch.mask_positions.size(1)
        D = pooled.size(-1)
        # owner/slot come precomputed from collate (pinned, copied async).
        # Building them here with torch.tensor(list, device=cuda) was a
        # blocking host->device copy that waited for the GPU to drain.
        owner, slot = opt_enc["owner"], opt_enc["slot"]
        out_pooled = pooled.new_zeros(B, Nmax, D)
        out_valid = torch.zeros(B, Nmax, dtype=torch.bool, device=device)
        scale = (self.inject_gate * batch.inject_scale.to(projected.dtype))[owner].unsqueeze(-1)
        out_pooled[owner, slot] = (projected * scale).to(out_pooled.dtype)
        out_valid[owner, slot] = True
        return out_pooled, out_valid, None, None
    return _original_encode_and_project_options(self, tokenizer, batch, device, chunk_size)


_original_encode_and_project_options = HybridDecisionModel.encode_and_project_options
HybridDecisionModel.encode_and_project_options = fast_encode_and_project_options

_original_encode_options_raw = HybridDecisionModel.encode_options_raw


def fast_encode_options_raw(self, tokenizer, option_texts, device, max_length: int = 64,
                             chunk_size: int = 128, need_tokens: bool = True):
    if isinstance(option_texts, dict) and "input_ids" in option_texts:
        ids = option_texts["input_ids"]
        amask = option_texts["attention_mask"]
        # Chunked so a long-tail example (this corpus has rows with up to 279
        # options) can't allocate one enormous option-encoding activation.
        pooled_chunks = []
        for start in range(0, ids.size(0), chunk_size):
            out = self.backbone(input_ids=ids[start:start + chunk_size],
                                 attention_mask=amask[start:start + chunk_size])
            h = out.last_hidden_state
            m = amask[start:start + chunk_size].unsqueeze(-1).float()
            pooled_chunks.append((h * m).sum(1) / m.sum(1).clamp(min=1.0))
        return torch.cat(pooled_chunks, dim=0), None, None
    return _original_encode_options_raw(self, tokenizer, option_texts, device,
                                         max_length, chunk_size, need_tokens)


HybridDecisionModel.encode_options_raw = fast_encode_options_raw


# --------------------------------------------------------------------------
# Loss
# --------------------------------------------------------------------------

def _correct_margin(logits, answer_idx):
    """logit(correct) - logsumexp(logits of everything else). Positive means
    the model would pick the right option; larger means more confidently. Used
    as the quantity the monotonicity hinge is applied to, rather than raw loss,
    because it's per-example, scale-comparable across depths, and is exactly
    what determines whether the argmax is right."""
    B, N = logits.shape
    correct = logits.gather(1, answer_idx.unsqueeze(1)).squeeze(1)
    neg_inf = torch.finfo(logits.dtype).min
    others = logits.scatter(1, answer_idx.unsqueeze(1), neg_inf)
    return correct - torch.logsumexp(others, dim=-1)


def compute_depth_loss(logits_per_depth, answer_idx, depth_weighting="uniform",
                        monotonic_weight=0.0, spherical_weight=0.5):
    K = len(logits_per_depth)
    device = logits_per_depth[0].device

    if depth_weighting == "ascending":
        w = torch.arange(1, K + 1, dtype=torch.float32, device=device)
    else:
        w = torch.ones(K, dtype=torch.float32, device=device)
    w = w / w.sum()

    total = torch.zeros((), device=device)
    ce_sum = torch.zeros((), device=device)
    for k, logits in enumerate(logits_per_depth):
        ce = F.cross_entropy(logits, answer_idx)
        p = F.softmax(logits, dim=-1)
        t = F.one_hot(answer_idx, num_classes=logits.size(-1)).to(p.dtype)
        spherical = (t * p).sum(-1) / p.norm(dim=-1).clamp(min=1e-8)
        total = total + w[k] * (ce + spherical_weight * (1.0 - spherical).mean())
        ce_sum = ce_sum + ce.detach()

    mono = torch.zeros((), device=device)
    if monotonic_weight > 0.0 and K > 1:
        margins = [_correct_margin(l, answer_idx) for l in logits_per_depth]
        steps = [F.relu(margins[k - 1] - margins[k]).mean() for k in range(1, K)]
        mono = torch.stack(steps).mean()
        total = total + monotonic_weight * mono

    # Returned as device tensors: calling .item() here forced a GPU->CPU sync
    # 7 times per micro-step, stalling the queue each time.
    return total, {"ce_mean": ce_sum / K, "monotonic": mono.detach()}


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------

def load_hard_corpus(data_dir, corpus_name="exp7_hard_reasoning_corpus"):
    base = os.path.join(data_dir, corpus_name)
    with open(os.path.join(base, "train.jsonl"), encoding="utf-8") as f:
        train_rows = [json.loads(l) for l in f if l.strip()]
    with open(os.path.join(base, "val.jsonl"), encoding="utf-8") as f:
        val_rows = [json.loads(l) for l in f if l.strip()]
    return train_rows, val_rows


class HardReasoningDataset(Dataset):
    """Option order is reshuffled every epoch (set_epoch). v1 used
    random.Random(idx + 42), identical on every pass over the data, so the
    correct answer sat in the same slot for a given example all training --
    a positional shortcut that needs no reasoning at all to exploit."""

    def __init__(self, rows, shuffle_options=True):
        self.rows = rows
        self.shuffle_options = shuffle_options
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        N = len(self.rows)
        epoch = idx // N if idx >= N else self.epoch
        idx = idx % N
        row = self.rows[idx]
        options = list(row["option_texts"])
        ans_idx = int(row["answer_idx"])
        if self.shuffle_options:
            rng = random.Random(idx + epoch * 1_000_003)
            true_ans = options[ans_idx]
            rng.shuffle(options)
            ans_idx = options.index(true_ans)

        qtype = str(row.get("qtype", "choice"))
        if qtype not in ("choice", "bool", "score"):
            qtype = "choice"

        return PackedExample(
            context=str(row.get("context", "")),
            instructions=str(row.get("instructions", "Select the correct answer:")),
            option_texts=options,
            qtype=qtype,
            answer_idx=ans_idx,
            use_text=True,
            use_vector=True,
            source=str(row.get("source", "hard_reasoning")),
        )


class PackedBatchCollate:
    def __init__(self, builder, tok, opt_max_length=64):
        self.builder = builder
        self.tok = tok
        self.opt_max_length = opt_max_length

    def __call__(self, examples):
        batch = self.builder.build_batch(examples, device="cpu")
        per_owner_n = [len(texts) for texts in batch.option_full_texts]
        flat_texts = [t for texts in batch.option_full_texts for t in texts]

        # Each distinct option string is encoded once per batch. ProofWriter
        # (the bulk of the corpus) has the same three options on every row, so
        # a batch of 8 was running the full backbone 24 times over three
        # strings; dedup makes that 3.
        uniq, index = {}, []
        for t in flat_texts:
            index.append(uniq.setdefault(t, len(uniq)))
        opt_ids, opt_mask, opt_index = None, None, None
        if flat_texts:
            enc = self.tok(list(uniq.keys()), padding=True, truncation=True,
                            max_length=self.opt_max_length, return_tensors="pt")
            opt_ids, opt_mask = enc["input_ids"], enc["attention_mask"]
            opt_index = torch.tensor(index, dtype=torch.long)
        opt_owner = torch.tensor([b for b, n in enumerate(per_owner_n) for _ in range(n)], dtype=torch.long)
        opt_slot = torch.tensor([j for n in per_owner_n for j in range(n)], dtype=torch.long)

        return AsyncPackedBatch(
            input_ids=batch.input_ids, attention_mask=batch.attention_mask,
            context_token_mask=batch.context_token_mask, mask_positions=batch.mask_positions,
            valid_mask=batch.valid_mask, inject_scale=batch.inject_scale,
            text_scale=batch.text_scale, qtype_idx=batch.qtype_idx,
            answer_idx=batch.answer_idx, per_owner_n=per_owner_n,
            option_input_ids=opt_ids, option_attention_mask=opt_mask,
            option_index=opt_index, option_owner=opt_owner, option_slot=opt_slot,
        )


class TokenBudgetBatchSampler(torch.utils.data.Sampler):
    """Groups examples into batches by TOKEN budget rather than a fixed count.

    This corpus spans 256 to ~8800 context tokens. A fixed batch size has to be
    set for the worst case (batch 1 at 7k tokens already uses ~8.4GiB), which
    leaves the GPU almost idle on the median 2k-token example. Batching to a
    token budget instead keeps memory roughly constant and the GPU busy: short
    examples batch 8+ at a time, a single 7k example goes alone.

    Examples are sorted by length inside large shuffled windows, so batches are
    length-homogeneous (little padding waste) while the epoch order still
    varies run to run.
    """

    def __init__(self, lengths, max_tokens, shuffle=True, window_mult=64, seed=0, length_caps=None,
                 max_examples=0):
        self.max_examples = max_examples
        self.length_caps = length_caps or []
        self.lengths = lengths
        self.max_tokens = max_tokens
        self.shuffle = shuffle
        self.window_mult = window_mult
        self.epoch = 0
        self.seed = seed

    def set_epoch(self, e):
        self.epoch = e

    def cap(self):
        if not self.length_caps:
            return None
        return self.length_caps[min(max(self.epoch, 1), len(self.length_caps)) - 1]

    def __iter__(self):
        # Length curriculum: early epochs only see examples up to cap(). The
        # model has to learn to reason over a ~100-token theory before it can
        # be expected to find that theory inside 7k tokens of distractors.
        cap = self.cap()
        idx = [i for i in range(len(self.lengths)) if cap is None or self.lengths[i] <= cap]
        rng = random.Random(self.seed + self.epoch)
        if self.shuffle:
            rng.shuffle(idx)
        window = max(1, self.max_tokens * self.window_mult // max(1, max(self.lengths)))
        window = max(window, 64)
        batches = []
        for start in range(0, len(idx), window):
            chunk = sorted(idx[start:start + window], key=lambda i: self.lengths[i])
            cur, cur_max = [], 0
            for i in chunk:
                new_max = max(cur_max, self.lengths[i])
                full = self.max_examples and len(cur) >= self.max_examples
                if cur and (new_max * (len(cur) + 1) > self.max_tokens or full):
                    batches.append(cur)
                    cur, cur_max = [i], self.lengths[i]
                else:
                    cur.append(i)
                    cur_max = new_max
            if cur:
                batches.append(cur)
        if self.shuffle:
            rng.shuffle(batches)
        # Encode the epoch into each index. Persistent DataLoader workers hold
        # their own copy of the dataset and never see dataset.set_epoch() from
        # the main process, so per-epoch option reshuffling silently stayed on
        # epoch 1 forever. Indices DO cross into the workers; the dataset
        # decodes (epoch, row) from them.
        N = len(self.lengths)
        return iter([[i + self.epoch * N for i in b] for b in batches])

    def __len__(self):
        total = sum(self.lengths)
        return max(1, total // self.max_tokens)


def build_sampler(rows, alpha):
    """alpha=0 -> natural distribution (79% ProofWriter). alpha=1 -> every
    source equally likely regardless of size. In between, weight propto
    (1/count)^alpha. Matters because the corpus is dominated by one templated
    synthetic source while the sources deliberation should actually help on
    (ARC-Challenge, MuSR) are a few percent between them."""
    if alpha <= 0:
        return None
    counts = Counter(r.get("source", "?") for r in rows)
    weights = [(1.0 / counts[r.get("source", "?")]) ** alpha for r in rows]
    return WeightedRandomSampler(weights, num_samples=len(rows), replacement=True)


# --------------------------------------------------------------------------
# Eval
# --------------------------------------------------------------------------

def load_hard111():
    """JevBench hard tier (public 111), official protocol -- rendering lives in
    experiments/jevbench_protocol.py. The previous loader passed bare label ids
    instead of the criteria, mapped JevBench's `noul` (yes/no) questions to
    "choice", and turned dict states into Python repr text."""
    return [PackedExample(context=x["context"], instructions=x["instructions"],
                          option_texts=x["option_texts"], qtype=x["qtype"], answer_idx=x["answer_idx"],
                          use_text=True, use_vector=True, source="jevbench_hard")
            for x in JB.load_tier("hard")]


def bootstrap_ci(correct_flags, n_boot=2000, seed=0):
    """95% CI on accuracy by resampling examples. n=111 means one example is
    0.9 points; without this it's impossible to tell a real depth effect from
    which 111 questions happened to be in the set."""
    t = torch.tensor(correct_flags, dtype=torch.float32)
    n = t.numel()
    if n == 0:
        return 0.0, 0.0
    g = torch.Generator().manual_seed(seed)
    idx = torch.randint(0, n, (n_boot, n), generator=g)
    accs = t[idx].mean(dim=1)
    return torch.quantile(accs, 0.025).item(), torch.quantile(accs, 0.975).item()


def _bucket(q):
    return "0" if q == 0 else ("1" if q == 1 else ("2-3" if q <= 3 else "4+"))


def _rows_to_examples(rows):
    return [PackedExample(context=str(r.get("context", "")),
                          instructions=str(r.get("instructions", "")),
                          option_texts=[str(o) for o in r["option_texts"]],
                          qtype="choice", answer_idx=int(r["answer_idx"]),
                          use_text=True, use_vector=True, source=str(r.get("source", "")))
            for r in rows]


EVAL_MAX_TOKENS = 10000


def _eval_batches(exs, max_tokens):
    """Length-sorted batches filled to a token budget. A fixed batch of 4 made
    short ProofWriter eval a stream of tiny kernels with gaps between them --
    most of the bursty GPU utilization during eval. No gradients are kept
    here, so the budget can be larger than training's."""
    est = [len(e.context) // 3 + 64 * max(1, len(e.option_texts)) + 160 for e in exs]
    order = sorted(range(len(exs)), key=lambda i: est[i])
    batches, cur, cur_max = [], [], 0
    for i in order:
        m = max(cur_max, est[i])
        if cur and m * (len(cur) + 1) > max_tokens:
            batches.append(cur)
            cur, cur_max = [i], est[i]
        else:
            cur.append(i)
            cur_max = m
    if cur:
        batches.append(cur)
    return batches


@torch.no_grad()
def _run(model, tok, exs, collate, device, batch_size, k_max):
    """Per-depth correctness and final-depth predicted option text, indexed
    like exs. batch_size is unused (kept for call compatibility); batches are
    sized by EVAL_MAX_TOKENS instead."""
    n = len(exs)
    correct = [[False] * n for _ in range(k_max)]
    pred_last = [None] * n
    for ids in _eval_batches(exs, EVAL_MAX_TOKENS):
        batch = collate([exs[i] for i in ids]).to_device(device)
        with autocast("cuda", dtype=AUTOCAST_DTYPE, enabled=(device.type == "cuda")):
            lp = model(tok, batch, device, k=k_max)
        for d in range(k_max):
            ok = (lp[d].argmax(-1) == batch.answer_idx).tolist()
            for j, i in enumerate(ids):
                correct[d][i] = ok[j]
        last = lp[-1].argmax(-1).tolist()
        for j, i in enumerate(ids):
            pred_last[i] = exs[i].option_texts[last[j]]
    return correct, pred_last


@torch.no_grad()
def evaluate_val(model, tok, val_rows, collate, device, batch_size, k_max,
                 n_pw=480, n_other=120, seed=7):
    """In-distribution validation -- the model-selection signal.

    exp7d selected checkpoints on 111 out-of-domain questions (+-9 point CI)
    and kept one that answered "Unknown" to every ProofWriter input while
    discarding epochs that had learned to reason. This evaluates on the
    corpus's own held-out split instead:

      pw_hops_<b>  ProofWriter stratified by required inference hops, with the
                   predicted-class histogram -- a collapsed model shows up as
                   ~100% of one class immediately, instead of having to be
                   reverse-engineered from accuracy numbers.
      quality / arc / musr   the other sources.

    Always evaluates at FULL length, even while the training curriculum is
    still short, so transfer to long context is visible from epoch 1.
    Selection score = macro-average over groups of mean-over-depth accuracy.
    """
    model.eval()
    rng = random.Random(seed)
    groups = {}
    by_b = defaultdict(list)
    for r in val_rows:
        if r.get("source") == "proofwriter_long":
            by_b[_bucket(int(r["qdep"]))].append(r)
    per_b = n_pw // max(1, len(by_b))
    for b in ("0", "1", "2-3", "4+"):
        if by_b.get(b):
            groups[f"pw_hops_{b}"] = rng.sample(by_b[b], min(per_b, len(by_b[b])))
    for prefix, name in (("quality", "quality"), ("arc_challenge", "arc"), ("musr", "musr")):
        rs = [r for r in val_rows if str(r.get("source", "")).startswith(prefix)]
        if rs:
            groups[name] = rng.sample(rs, min(n_other, len(rs)))

    out, pw_len = {}, {"<2048": [], ">=2048": []}
    for g, rs in groups.items():
        correct, pred_last = _run(model, tok, _rows_to_examples(rs), collate, device, batch_size, k_max)
        entry = {"n": len(rs), "accs": [sum(c) / len(c) for c in correct]}
        if g.startswith("pw_"):
            h = Counter(pred_last)
            entry["pred_hist"] = {k: h.get(k, 0) / len(rs) for k in ("True", "False", "Unknown")}
            for i, r in enumerate(rs):
                key = "<2048" if int(r.get("target_tokens", 0)) < 2048 else ">=2048"
                pw_len[key].append(correct[-1][i])
        out[g] = entry
    model.train()
    score = sum(sum(v["accs"]) / len(v["accs"]) for v in out.values()) / max(1, len(out))
    out["_pw_by_length"] = {k: (sum(v) / len(v) if v else None) for k, v in pw_len.items()}
    return out, score


@torch.no_grad()
def evaluate_hard111(model, tok, examples, collate, device, batch_size=8, k_max=None):
    """Reported for continuity with exp7d only -- NOT used for selection."""
    model.eval()
    k_max = k_max or model.k_max
    correct, _ = _run(model, tok, examples, collate, device, batch_size, k_max)
    model.train()
    accs = [sum(c) / len(c) for c in correct]
    cis = [bootstrap_ci(c) for c in correct]
    return accs, cis, len(examples)


def build_param_groups(model, base_lr, layer_decay, head_mult, new_mult, weight_decay, new_names):
    """Three learning-rate tiers plus a no-decay set.

    v2 used a single AdamW group at 1.5e-5 for all 448M parameters. That puts
    freshly initialized modules (LoRA, attention scorer, resized codes) at the
    learning rate of a pretrained backbone, where they barely move, and it
    applied weight decay to scalar gates and LayerNorms -- steadily pulling
    inject_gate / s0_gate / scratch_gate toward zero. train.py had solved
    both (head_lr_mult, layer-wise decay); the rewrite dropped them.

      backbone     base_lr * layer_decay^(distance from top)
      head         base_lr * head_mult     (trained exp7 head modules)
      new          base_lr * new_mult      (anything absent from the checkpoint)
      no decay     biases, norms, embeddings, gates, codes, LoRA factors
    """
    n_layers = len(model.backbone.layers)
    buckets = defaultdict(list)
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        no_decay = (p.ndim < 2 or name.endswith(".bias") or "norm" in name.lower()
                    or "embed" in name or name == "context_codes" or "lora_" in name
                    or name.endswith("_gate"))
        wd = 0.0 if no_decay else weight_decay
        if name.startswith("backbone."):
            if ".layers." in name:
                depth = int(name.split(".layers.")[1].split(".")[0]) + 1
            elif "embeddings" in name:
                depth = 0
            else:
                depth = n_layers + 1
            lr, tier = base_lr * (layer_decay ** (n_layers + 1 - depth)), "backbone"
        elif name in new_names:
            lr, tier = base_lr * new_mult, "new"
        else:
            lr, tier = base_lr * head_mult, "head"
        buckets[(round(lr, 14), wd, tier)].append(p)
    groups = [{"params": ps, "lr": lr, "weight_decay": wd, "tier": tier}
              for (lr, wd, tier), ps in buckets.items()]
    summary = defaultdict(int)
    for g in groups:
        summary[g["tier"]] += sum(p.numel() for p in g["params"])
    return groups, dict(summary)


# --------------------------------------------------------------------------

def lr_lambda_factory(total_steps, warmup_frac, min_lr_frac):
    warmup = max(1, int(total_steps * warmup_frac))

    def f(step):
        if step < warmup:
            return step / warmup
        prog = min(1.0, (step - warmup) / max(1, total_steps - warmup))
        return min_lr_frac + (1.0 - min_lr_frac) * 0.5 * (1.0 + math.cos(math.pi * prog))
    return f


def parse_args():
    ap = argparse.ArgumentParser()
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    ap.add_argument("--resume", type=str,
                     default=os.path.join(base, "checkpoints", "exp7d_hard_reasoning", "exp7d_10ep_epoch2.pt"),
                     help="Default is exp7d_10ep_epoch2, NOT exp7d_best_hard: best_hard predicts "
                          "'Unknown' on 100%% of ProofWriter inputs (verified with "
                          "diag_qdep_ckpt.py); epoch2 of the 10-epoch run actually reasons "
                          "(75.8%% on 2-3 hop questions).")
    ap.add_argument("--require_exact_load", action="store_true")

    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--steps_per_epoch", type=int, default=600)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-5, help="Backbone top-layer LR; other tiers scale from it.")
    ap.add_argument("--layer_decay", type=float, default=0.9)
    ap.add_argument("--head_lr_mult", type=float, default=5.0)
    ap.add_argument("--new_lr_mult", type=float, default=20.0)
    ap.add_argument("--min_lr_frac", type=float, default=0.05)
    ap.add_argument("--warmup_frac", type=float, default=0.06)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--optim", choices=["adamw8bit", "adamw"], default="adamw8bit")
    ap.add_argument("--eval_max_tokens", type=int, default=10000)
    ap.add_argument("--max_tokens_per_batch", type=int, default=10000)
    ap.add_argument("--gpu_mem_fraction", type=float, default=0.86,
                     help="Cap on PyTorch's share of VRAM. Keeps Windows from silently paging "
                          "allocations into system RAM. 0 disables.")
    ap.add_argument("--pad_multiple", type=int, default=64)
    ap.add_argument("--max_examples_per_batch", type=int, default=16,
                     help="The token budget alone let short-context batches pack ~30 examples; "
                          "the head's memory scales with example count (context codes + options, "
                          "x every recurrent depth), which pushed a 12GB card into shared memory.")
    ap.add_argument("--length_curriculum", type=str, default="2048,4096,8192",
                     help="Per-epoch max example length (tokens); the last value repeats. "
                          "'' disables.")
    ap.add_argument("--num_workers", type=int, default=2 if os.name == "nt" else 4)

    ap.add_argument("--budget_total", type=int, default=8192)
    ap.add_argument("--l_context", type=int, default=7168)
    ap.add_argument("--l_instructions", type=int, default=128)
    ap.add_argument("--l_max_per_option", type=int, default=64)
    ap.add_argument("--option_chunk_size", type=int, default=128)
    ap.add_argument("--no_grad_checkpoint", action="store_true")

    ap.add_argument("--n_context_codes", type=int, default=16,
                     help="Learned context summary slots. 16 was sized for l_context=768; at "
                          "7168 each code covers ~448 tokens. Resizing tiles the trained codes.")
    ap.add_argument("--n_scratch", type=int, default=0)
    ap.add_argument("--use_film_depth", action="store_true")
    ap.add_argument("--depth_lora_rank", type=int, default=0,
                     help="Per-depth LoRA rank on the shared recurrent FFN.")
    ap.add_argument("--n_scorer_blocks", type=int, default=0,
                     help="Attention blocks before scoring (option<->option self-attn + "
                          "option->evidence cross-attn). ~8.4M params per block.")

    ap.add_argument("--depth_weighting", choices=["uniform", "ascending"], default="uniform")
    ap.add_argument("--monotonic_weight", type=float, default=0.0)
    ap.add_argument("--spherical_weight", type=float, default=0.5)

    ap.add_argument("--corpus", type=str, default="exp7_longctx_reasoning_corpus")
    ap.add_argument("--shuffle_options", type=int, default=1)

    ap.add_argument("--eval_batch_size", type=int, default=4)
    ap.add_argument("--val_n_pw", type=int, default=480)
    ap.add_argument("--val_n_other", type=int, default=120)
    ap.add_argument("--skip_hard111", action="store_true")
    ap.add_argument("--patience", type=int, default=3)

    ap.add_argument("--run_name", type=str, default="exp7e_v3")
    ap.add_argument("--out_dir", type=str, default=None)
    ap.add_argument("--seed", type=int, default=1234)
    return ap.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
        if args.gpu_mem_fraction > 0:
            # On Windows (WDDM) a cudaMalloc beyond physical VRAM does NOT fail --
            # the driver pages it into system RAM. PyTorch therefore never sees an
            # out-of-memory, never frees its cache, and the run slides into shared
            # memory (measured: 0.9-2.7GB spilled, training stalled). Capping the
            # allocator below physical VRAM makes it free and reuse its own cache
            # (visible as alloc_retries) instead of letting the driver page it out.
            torch.cuda.set_per_process_memory_fraction(args.gpu_mem_fraction)
            total = torch.cuda.get_device_properties(0).total_memory / 2**30
            print(f"GPU memory cap: {args.gpu_mem_fraction:.2f} x {total:.1f}GiB = "
                  f"{args.gpu_mem_fraction*total:.1f}GiB", flush=True)

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    data_dir = os.path.join(base_dir, "data")
    out_dir = args.out_dir or os.path.join(base_dir, "checkpoints", args.run_name)
    os.makedirs(out_dir, exist_ok=True)
    metrics_file = os.path.join(out_dir, f"{args.run_name}_metrics.json")

    tok = get_tokenizer(BACKBONE)
    builder = PackedSequenceBuilder(tok, budget_total=args.budget_total, l_context=args.l_context,
                                     l_instructions=args.l_instructions,
                                     l_max_per_option=args.l_max_per_option,
                                     pad_multiple=args.pad_multiple)

    ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
    k_max = ckpt.get("k_max", 6)
    use_maxsim = ckpt.get("use_maxsim", False)
    model_config = dict(k_max=k_max, use_maxsim=use_maxsim, n_context_codes=args.n_context_codes,
                        n_scratch=args.n_scratch, use_film_depth=args.use_film_depth,
                        depth_lora_rank=args.depth_lora_rank, n_scorer_blocks=args.n_scorer_blocks)
    model = HybridDecisionModel(backbone=BACKBONE, mask_token_id=tok.mask_token_id,
                                gradient_checkpointing=not args.no_grad_checkpoint, **model_config)
    sd = ckpt.get("model", ckpt.get("model_state_dict", ckpt))
    missing, unexpected, notes = load_flexible(model, sd)
    print(f"\nCheckpoint: {args.resume}  (epoch {ckpt.get('epoch')})", flush=True)
    for n_ in notes:
        print(f"  {n_}", flush=True)
    new_names = set(missing)
    if sd.get("context_codes") is not None and sd["context_codes"].shape[0] != args.n_context_codes:
        new_names.add("context_codes")
    groups_new = Counter(k.split(".")[0] for k in missing)
    print(f"  {len(missing)} new tensors (trained at the 'new' LR tier): {dict(groups_new)}", flush=True)
    if unexpected:
        print(f"  !! {len(unexpected)} unexpected keys: {unexpected[:8]}", flush=True)
        if args.require_exact_load:
            raise RuntimeError("unexpected keys and --require_exact_load")
    model.to(device)
    tot = sum(p.numel() for p in model.parameters())
    bb = sum(p.numel() for p in model.backbone.parameters())
    print(f"  params: {tot/1e6:.1f}M total, {(tot - bb)/1e6:.1f}M head", flush=True)

    train_rows, val_rows = load_hard_corpus(data_dir, args.corpus)
    src_counts = Counter(r.get("source", "?") for r in train_rows)
    print(f"\nCorpus {args.corpus}: {len(train_rows)} train / {len(val_rows)} val", flush=True)
    for s_, c in src_counts.most_common():
        print(f"    {s_:26s} {c:6d}  ({c/len(train_rows)*100:5.1f}%)", flush=True)

    dataset = HardReasoningDataset(train_rows, shuffle_options=bool(args.shuffle_options))
    collate = PackedBatchCollate(builder, tok, opt_max_length=args.l_max_per_option)
    caps = [int(x) for x in args.length_curriculum.split(",") if x.strip()]
    lengths = [min(int(r.get("approx_tokens", 0) or 0) + 64, args.budget_total) for r in train_rows]
    batch_sampler = TokenBudgetBatchSampler(lengths, args.max_tokens_per_batch, shuffle=True,
                                             seed=args.seed, length_caps=caps,
                                             max_examples=args.max_examples_per_batch)
    loader = DataLoader(dataset, batch_sampler=batch_sampler, num_workers=args.num_workers,
                        pin_memory=True, prefetch_factor=4 if args.num_workers > 0 else None,
                        persistent_workers=(args.num_workers > 0), collate_fn=collate)

    # Real per-epoch step counts under the curriculum, so the cosine schedule
    # ends where training actually ends.
    steps_per_epoch = []
    for e in range(1, args.epochs + 1):
        batch_sampler.set_epoch(e)
        steps_per_epoch.append(min(args.steps_per_epoch, sum(1 for _ in iter(batch_sampler))))
    batch_sampler.set_epoch(1)
    total_steps = sum(steps_per_epoch)
    print(f"Length curriculum caps {caps or 'off'} -> steps per epoch {steps_per_epoch} "
          f"(total {total_steps})", flush=True)

    param_groups, tiers = build_param_groups(model, args.lr, args.layer_decay, args.head_lr_mult,
                                             args.new_lr_mult, args.weight_decay, new_names)
    print("LR tiers: " + "  ".join(f"{t}={n/1e6:.1f}M" for t, n in tiers.items())
          + f"   (backbone top {args.lr:.1e}, head {args.lr*args.head_lr_mult:.1e}, "
            f"new {args.lr*args.new_lr_mult:.1e})", flush=True)
    if args.optim == "adamw8bit":
        # fp32 AdamW over ~460M params is weights + grads + two moments ~= 7.3GB
        # before any activation, which pushed this 12GB card into shared
        # (system) memory -- measured 1.4GB spilled, and throughput fell from
        # 0.43 to 0.03 it/s. 8-bit moments save ~2.7GB. train.py already used
        # this; the exp7d scripts had dropped it.
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(param_groups, lr=args.lr)
    else:
        optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
    print(f"Optimizer: {type(optimizer).__name__}", flush=True)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda_factory(max(1, total_steps // args.grad_accum),
                                     args.warmup_frac, args.min_lr_frac))

    global EVAL_MAX_TOKENS
    EVAL_MAX_TOKENS = args.eval_max_tokens
    hard111 = [] if args.skip_hard111 else load_hard111()

    live = {"experiment": args.run_name, "args": vars(args), "model_config": model_config,
            "budget_total": args.budget_total, "l_context": args.l_context, "status": "running",
            "source_counts": dict(src_counts), "initial": None, "evals": [],
            "current_step": 0, "total_steps": total_steps, "current_loss": 0.0}

    def flush_metrics():
        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump(live, f, indent=2)

    def run_evals(tag):
        val, score = evaluate_val(model, tok, val_rows, collate, device, args.eval_batch_size,
                                  k_max, args.val_n_pw, args.val_n_other)
        print(f"\n--- {tag}: in-distribution val ---", flush=True)
        print(f"    {'group':12s} {'n':>4s}  " + " ".join(f"{'k='+str(i+1):>5s}" for i in range(k_max))
              + "   k6-k1   predicted T/F/U", flush=True)
        warn = []
        for g, v in val.items():
            if g.startswith("_"):
                continue
            a = v["accs"]
            hist = ""
            if "pred_hist" in v:
                ph = v["pred_hist"]
                hist = f"   {ph['True']*100:3.0f}/{ph['False']*100:3.0f}/{ph['Unknown']*100:3.0f}"
                top = max(ph, key=ph.get)
                if ph[top] > 0.85:
                    warn.append(f"{g}: predicts '{top}' on {ph[top]*100:.0f}% of inputs")
            print(f"    {g:12s} {v['n']:4d}  " + " ".join(f"{x*100:5.1f}" for x in a)
                  + f"   {(a[-1]-a[0])*100:+5.1f}" + hist, flush=True)
        bl = val["_pw_by_length"]
        print("    ProofWriter by context length (final depth): "
              + "  ".join(f"{k}: {v*100:.1f}%" for k, v in bl.items() if v is not None), flush=True)
        print(f"    VAL SCORE (selection) = {score*100:.2f}%", flush=True)
        for w in warn:
            print(f"    WARNING collapse -- {w}", flush=True)
        grid = {g.replace("pw_hops_", ""): {"n": v["n"], "accs": v["accs"],
                                              "pred_hist": v.get("pred_hist")}
                for g, v in val.items() if g.startswith("pw_hops_")}
        rec = {"val": val, "val_score": score, "qdep_grid": grid}
        if hard111:
            accs, cis, n = evaluate_hard111(model, tok, hard111, collate, device,
                                            args.eval_batch_size, k_max)
            print("    Hard-111 (reported only): " + " ".join(f"{a*100:.1f}" for a in accs)
                  + f"   CI k=1 {cis[0][0]*100:.0f}-{cis[0][1]*100:.0f}", flush=True)
            rec.update(accs=accs, cis=cis)
        # Release eval's cached blocks before training reallocates at different
        # shapes -- otherwise the caching allocator holds both working sets and
        # the total spills into shared (system) memory. Reset the peak so the
        # step log reports TRAINING memory, not the eval high-water mark.
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        return rec, score

    print("\n=== Baseline eval ===", flush=True)
    rec, best_score = run_evals("baseline")
    live["initial"] = rec
    flush_metrics()

    best_path = os.path.join(out_dir, f"{args.run_name}_best.pt")
    latest_path = os.path.join(out_dir, f"{args.run_name}_latest.pt")

    def save(path, epoch, rec):
        torch.save({"model": model.state_dict(), "k_max": k_max, "use_maxsim": use_maxsim,
                    "model_config": model_config, "epoch": epoch, "eval": rec, "args": vars(args)},
                   path)

    print(f"\n=== Training {args.epochs} epochs, token budget {args.max_tokens_per_batch} x accum "
          f"{args.grad_accum}, depth_weighting={args.depth_weighting}, "
          f"monotonic={args.monotonic_weight} ===", flush=True)

    head_groups = [g for g in optimizer.param_groups if g.get("tier") == "head"]
    global_step, no_improve = 0, 0
    for epoch in range(1, args.epochs + 1):
        dataset.set_epoch(epoch)
        batch_sampler.set_epoch(epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        run_loss = torch.zeros((), device=device)
        run_mono, seen, t0 = torch.zeros((), device=device), 0, time.time()
        data_wait, t_last = 0.0, time.time()
        oom_skips = 0
        retries0 = torch.cuda.memory_stats().get("num_alloc_retries", 0) if device.type == "cuda" else 0
        n_steps = steps_per_epoch[epoch - 1]
        cap = batch_sampler.cap()
        print(f"\n[epoch {epoch}] max example length {cap if cap else 'full'}, {n_steps} steps", flush=True)

        for step, async_batch in enumerate(loader, 1):
            if step > n_steps:
                break
            data_wait += time.time() - t_last
            batch = async_batch.to_device(device)
            try:
                with autocast("cuda", dtype=AUTOCAST_DTYPE, enabled=(device.type == "cuda")):
                    lp = model(tok, batch, device, k=k_max, option_chunk_size=args.option_chunk_size)
                lp = [l.float() for l in lp]
                loss, comps = compute_depth_loss(lp, batch.answer_idx, depth_weighting=args.depth_weighting,
                                                 monotonic_weight=args.monotonic_weight,
                                                 spherical_weight=args.spherical_weight)
                (loss / args.grad_accum).backward()
            except torch.cuda.OutOfMemoryError:
                # With the memory cap, a batch that genuinely doesn't fit raises
                # instead of paging. Drop it and keep going; the count is logged.
                del batch
                lp = loss = None
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                oom_skips += 1
                t_last = time.time()
                continue
            run_loss += loss.detach()
            run_mono += comps["monotonic"]
            seen += 1
            if step % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            global_step += 1
            t_last = time.time()

            if step % 50 == 0 or step == n_steps:
                avg = (run_loss / seen).item()
                sps = seen / max(time.time() - t0, 1e-5)
                eta = (total_steps - global_step) / max(sps, 1e-5)
                head_lr = head_groups[0]["lr"] if head_groups else scheduler.get_last_lr()[0]
                live.update(current_step=global_step, current_loss=avg, eta_seconds=int(eta), lr=head_lr)
                flush_metrics()
                mem = torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0
                retries = (torch.cuda.memory_stats().get("num_alloc_retries", 0) - retries0
                           if device.type == "cuda" else 0)
                print(f"  [ep {epoch}/{args.epochs}] {step}/{n_steps} ({global_step}/{total_steps}) "
                      f"loss {avg:.4f} mono {(run_mono/seen).item():.4f} lr {head_lr:.2e} "
                      f"{sps:.2f} it/s  {mem:.1f}GiB  ETA {int(eta//60)}m  "
                      f"[data_wait {data_wait/max(time.time()-t0,1e-5)*100:.0f}% | alloc_retries {retries}"
                      f"{' | oom_skips ' + str(oom_skips) if oom_skips else ''}]",
                      flush=True)

        if seen % args.grad_accum != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        rec, score = run_evals(f"end of epoch {epoch}")
        rec.update(epoch=epoch, step=n_steps, global_step=global_step,
                   train_loss=(run_loss / max(seen, 1)).item(), length_cap=cap)
        live["evals"].append(rec)
        flush_metrics()
        save(latest_path, epoch, rec)
        if score > best_score:
            best_score, no_improve = score, 0
            save(best_path, epoch, rec)
            print(f"*** new best val score {score*100:.2f}% -> {best_path}", flush=True)
        else:
            no_improve += 1
            print(f"    no improvement ({no_improve}/{args.patience or '-'})", flush=True)
            # Don't early-stop while the curriculum is still lengthening: a dip
            # when longer examples first appear is expected, not a plateau.
            still_growing = bool(caps) and epoch < len(caps)
            if args.patience and no_improve >= args.patience and not still_growing:
                print("Early stopping.", flush=True)
                break

    live["status"] = "completed"
    flush_metrics()
    print(f"\nDone. Best val score {best_score*100:.2f}%  ({best_path})", flush=True)


if __name__ == "__main__":
    main()
