"""exp7c: learned halting + escalation (NOTES.md Sec 6).

Deliberately NOT trained jointly with the depth-conditioned model (unlike
ACT/PonderNet-style differentiable halting). NOTES.md's design is a frozen,
post-hoc, self-supervised classifier on top of an already-trained depth
trajectory p_1..p_K_max:

  features(k) = [top_prob(p_k), margin(p_k), entropy(p_k), N, k, KL(p_k||p_k-1)]
  halt_head:     features(k) -> P(more passes would not change the answer)
  escalate_head: features(k_halt) -> P(the halted-depth answer is wrong)

halt_head's target needs no new labels: halt=1 iff argmax(p_k)==argmax(p_K_max)
("if I'd kept going, would I have landed somewhere else"), computed from the
SAME forward pass. escalate_head's target uses the real answer_idx (it's
asking a different question -- not "did more passes agree with me" but "was
I actually right") and is calibrated via split conformal on a held-out set so
its flagged rate carries a distribution-free guarantee rather than a
hand-picked threshold.

Both heads are tiny MLPs over the 6-dim feature vector -- the whole point is
that they're cheap enough to run before every escalation decision, unlike
another full recurrent pass.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


FEATURE_NAMES = ["top_prob", "margin", "entropy", "n_options", "depth", "kl_prev"]
N_FEATURES = len(FEATURE_NAMES)


class HaltingHeads(nn.Module):
    def __init__(self, hidden_mult: int = 8, dropout: float = 0.1):
        super().__init__()
        hidden = N_FEATURES * hidden_mult
        self.halt_head = nn.Sequential(
            nn.LayerNorm(N_FEATURES), nn.Linear(N_FEATURES, hidden), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden, 1),
        )
        self.escalate_head = nn.Sequential(
            nn.LayerNorm(N_FEATURES), nn.Linear(N_FEATURES, hidden), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden, 1),
        )

    def forward(self, feats: torch.Tensor):
        """feats: (..., 6). Returns (halt_logit, escalate_logit), each (...,)."""
        return self.halt_head(feats).squeeze(-1), self.escalate_head(feats).squeeze(-1)


@torch.no_grad()
def compute_depth_features(logits_per_depth, valid_mask):
    """logits_per_depth: list of K (B, N) tensors, -inf at padding.
    Returns (feats_per_depth, probs_per_depth): each a list of K tensors,
    feats (B, 6), probs (B, N). No_grad -- this runs on top of an entirely
    frozen model, only HaltingHeads itself is ever trained."""
    n_count = valid_mask.sum(-1).float()  # (B,)
    probs_per_depth = [F.softmax(l, dim=-1) for l in logits_per_depth]

    feats_per_depth = []
    prev_p = None
    for k, p in enumerate(probs_per_depth):
        n_real = min(2, p.size(-1))
        top_vals = torch.topk(p, k=n_real, dim=-1).values
        top1 = top_vals[:, 0]
        top2 = top_vals[:, 1] if n_real > 1 else torch.zeros_like(top1)
        margin = top1 - top2
        logp = p.clamp_min(1e-12).log()
        entropy = -(p * logp).sum(-1)
        if prev_p is None:
            kl = torch.zeros_like(top1)
        else:
            kl = (p * (logp - prev_p.clamp_min(1e-12).log())).sum(-1)
        depth_idx = torch.full_like(top1, float(k + 1))
        feats_per_depth.append(torch.stack([top1, margin, entropy, n_count, depth_idx, kl], dim=-1))
        prev_p = p
    return feats_per_depth, probs_per_depth


@torch.no_grad()
def compute_targets(probs_per_depth, answer_idx):
    """halt_target(k) = 1 iff this depth already agrees with the K_max answer
    (self-supervised -- no ground truth needed). escalate_target(k) = 1 iff
    this depth's answer is actually wrong (needs answer_idx)."""
    final_argmax = probs_per_depth[-1].argmax(-1)
    halt_targets = [(p.argmax(-1) == final_argmax).float() for p in probs_per_depth]
    escalate_targets = [(p.argmax(-1) != answer_idx).float() for p in probs_per_depth]
    return halt_targets, escalate_targets


def halted_depth_and_prob(feats_per_depth, halting_heads: HaltingHeads, theta: float):
    """Simulates the actual inference-time policy: walk depths 1..K, stop at
    the first k where sigmoid(halt_head(feats_k)) >= theta (or at K_max if
    never confident). Returns (halted_depth_idx0 (B,) long, escalate_prob_at_halt (B,)).
    Vectorized over the batch even though different examples halt at different depths."""
    K = len(feats_per_depth)
    B = feats_per_depth[0].size(0)
    device = feats_per_depth[0].device
    halted = torch.zeros(B, dtype=torch.long, device=device)
    escalate_prob_at_halt = torch.zeros(B, device=device)
    still_running = torch.ones(B, dtype=torch.bool, device=device)
    for k in range(K):
        halt_logit, esc_logit = halting_heads(feats_per_depth[k])
        halt_prob = torch.sigmoid(halt_logit)
        esc_prob = torch.sigmoid(esc_logit)
        should_stop = still_running & ((halt_prob >= theta) | (k == K - 1))
        halted = torch.where(should_stop, torch.full_like(halted, k), halted)
        escalate_prob_at_halt = torch.where(should_stop, esc_prob, escalate_prob_at_halt)
        still_running = still_running & ~should_stop
    return halted, escalate_prob_at_halt


def compute_speedup_curve(feats_per_depth, probs_per_depth, halting_heads: HaltingHeads,
                           answer_idx: torch.Tensor, thetas=(0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99)):
    """Compute-vs-accuracy at a sweep of halting thresholds, on real labels
    (not the self-supervised halt target) -- this is the actual deliverable
    NOTES.md Sec 6 asks for: a curve to pick theta from, trading average
    depth used against accuracy actually achieved at that depth."""
    K = len(feats_per_depth)
    rows = []
    for theta in thetas:
        halted, _ = halted_depth_and_prob(feats_per_depth, halting_heads, theta)
        B = halted.size(0)
        idx = torch.arange(B, device=halted.device)
        chosen_pred = torch.stack([p.argmax(-1) for p in probs_per_depth], dim=1)[idx, halted]  # (B,)
        acc = (chosen_pred == answer_idx).float().mean().item()
        avg_depth = (halted.float() + 1).mean().item()
        rows.append({"theta": theta, "avg_depth": avg_depth, "accuracy": acc, "of_k_max": K})
    return rows


def calibrate_escalation_threshold(escalate_probs_wrong: torch.Tensor, alpha: float = 0.1):
    """Split conformal calibration (NOTES.md Sec 6): escalate_probs_wrong are
    the escalate_head outputs on calibration examples where the model's
    halted-depth answer was ACTUALLY wrong. Pick tau = the
    ceil((1-alpha)*(n+1))/n empirical quantile of these scores (the standard
    finite-sample split-conformal correction) so that, for future exchangeable
    data, at least (1-alpha) of truly-wrong answers get escalate_prob >= tau
    -- a distribution-free guarantee on the miss rate, not a hand-tuned cutoff.
    Returns (tau, n_calibration_wrong)."""
    n = escalate_probs_wrong.numel()
    if n == 0:
        return 0.5, 0  # no wrong examples in calibration split -- can't calibrate, fall back
    sorted_probs, _ = torch.sort(escalate_probs_wrong)
    # tau = sorted_probs[rank] (0-indexed), rank = floor(alpha*(n+1)) - 1, clamped into
    # [0, n-1] -- the standard finite-sample split-conformal quantile, chosen so that at
    # least ceil((1-alpha)*(n+1)) of the n+1 exchangeable scores (n seen + 1 future) fall
    # at or above tau, i.e. a future truly-wrong example is flagged with probability >= 1-alpha.
    rank = max(0, min(n - 1, math.floor(alpha * (n + 1)) - 1))
    tau = sorted_probs[rank].item()
    return tau, n
