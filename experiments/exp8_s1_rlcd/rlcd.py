"""RLCD-style objective for exp8: reinforcement learning for calibrated decisions.

What is actually known (checked 2026-09-24): TypeSafe names RLCD and states
its goal -- "decisions and calibrated probabilities", "higher confidence means
higher accuracy" -- but publishes no reward function, scoring rule, estimator,
data or code. The only public implementation of the idea is Convai's Laya
(experiments/LAYA_COMPARISON_REPORT.md Sec 3): the reward for a reported
distribution q is a strictly proper scoring rule,

    R(q, y) = log q_y + 0.5 * spherical(q, y) - 1.0 * RPS(q, y) [ordinal only]

optimized with a GRPO-style group-mean-baseline policy gradient. A strictly
proper rule is maximized in expectation ONLY by reporting the true belief --
that is the whole calibration mechanism.

The policy's action is the reported distribution itself. For single-turn
questions with a known answer, the expected reward J(theta) = E[R(q_theta, y)]
has an exact, zero-variance gradient (R is differentiable in q). A sampled
policy-gradient estimator targets the SAME J with added variance. So:

  --estimator exact    (default) exact policy gradient of J. Numerically this
                       is "minimize -R", i.e. proper-scoring-rule training.
  --estimator grpo     genuinely sampled RL, for comparison: the report is a
                       stochastic action q ~ Dirichlet(kappa * p_theta); G
                       reports per example; reward R(q_g, y); advantage =
                       (R_g - mean_g R) / std_g R; loss = -mean(A_g * log pi(q_g)).
                       Unbiased for a smoothed J that approaches the exact one
                       as kappa grows. Costs no extra forward passes.
  --estimator hybrid   exact + lambda * grpo.

Honest expectation: exact and grpo should converge to the same place, grpo
more slowly. Laya's multi-turn TD(lambda) bootstrapping is the part where RL
is genuinely needed; there is no multi-turn data here, so it isn't implemented.

Also here: calibration metrics (NLL, Brier, ECE, reliability bins) and
decision-utility curves (answer vs escalate), which are what "calibrated
decisions" buys downstream.
"""
import math

import torch
import torch.nn.functional as F

from s1_model import QTYPE_IDX

SCORE_Q = QTYPE_IDX["score"]


def proper_reward(logits, y, qtype_idx, spherical_w=0.5, rps_w=1.0):
    """Per-example reward components. logits (B,N) with -inf at padding."""
    logp = F.log_softmax(logits.float(), -1)
    p = logp.exp()
    log_score = logp.gather(1, y[:, None]).squeeze(1)
    spherical = p.gather(1, y[:, None]).squeeze(1) / p.norm(dim=-1).clamp(min=1e-8)
    t = F.one_hot(y, logits.size(-1)).to(p.dtype)
    rps = ((p.cumsum(-1) - t.cumsum(-1)) ** 2).sum(-1)
    is_ord = (qtype_idx == SCORE_Q).to(p.dtype)
    R = log_score + spherical_w * spherical - rps_w * rps * is_ord
    return R, {"log_score": log_score, "spherical": spherical, "rps": rps, "is_ordinal": is_ord}


def grpo_loss(logits, y, qtype_idx, valid, G=8, kappa=50.0, spherical_w=0.5, rps_w=1.0):
    """Sampled policy gradient over probability reports. Reports are drawn from
    Dirichlet(kappa * p) restricted to valid options; the reward is the same
    proper score as the exact estimator."""
    p = F.softmax(logits.float(), -1).clamp(min=1e-6) * valid
    p = p / p.sum(-1, keepdim=True)
    conc = (kappa * p).clamp(min=1e-3) * valid + (~valid) * 1e-3
    dist = torch.distributions.Dirichlet(conc)
    q = dist.rsample((G,)).detach()                        # (G,B,N) actions, no pathwise grad
    q = (q * valid).clamp(min=1e-8)
    q = q / q.sum(-1, keepdim=True)
    logq = q.log().masked_fill(~valid, float("-inf"))
    Rg = torch.stack([proper_reward(logq[g], y, qtype_idx, spherical_w, rps_w)[0] for g in range(G)])
    A = (Rg - Rg.mean(0, keepdim=True)) / Rg.std(0, keepdim=True).clamp(min=1e-6)
    logpi = dist.log_prob(q.clamp(min=1e-6) / q.clamp(min=1e-6).sum(-1, keepdim=True))  # (G,B)
    return -(A.detach() * logpi).mean(), Rg.mean(0).detach()


def objective(logits, y, qtype_idx, valid, estimator="exact", lam=0.5, **kw):
    R, comps = proper_reward(logits, y, qtype_idx)
    if estimator == "exact":
        loss_per_ex = -R
        loss = loss_per_ex.mean()
    elif estimator == "grpo":
        loss, _ = grpo_loss(logits, y, qtype_idx, valid, **kw)
        loss_per_ex = -R.detach()
    else:
        g, _ = grpo_loss(logits, y, qtype_idx, valid, **kw)
        loss_per_ex = -R
        loss = (-R).mean() + lam * g
    return loss, R.detach(), {k: v.detach() for k, v in comps.items()}, loss_per_ex.detach()


# --------------------------------------------------------------------------
# Calibration + decision metrics (on accumulated per-example records)
# --------------------------------------------------------------------------

def ece(conf, correct, n_bins=15):
    if not conf:
        return float("nan"), []
    bins = [[0, 0.0, 0.0] for _ in range(n_bins)]
    for c, k in zip(conf, correct):
        b = min(int(c * n_bins), n_bins - 1)
        bins[b][0] += 1
        bins[b][1] += c
        bins[b][2] += k
    n = len(conf)
    e = sum(abs(s_c / m - s_k / m) * m / n for m, s_c, s_k in bins if m)
    rel = [{"bin": i, "n": m, "conf": s_c / m, "acc": s_k / m} for i, (m, s_c, s_k) in enumerate(bins) if m]
    return e, rel


def summarize(records):
    """records: list of dicts with correct, conf (max prob), p_true, nll, brier,
    n_options, qtype, blind_easy, [ordinal: exp_level, true_level, rps]."""
    if not records:
        return {}
    n = len(records)
    acc = sum(r["correct"] for r in records) / n
    e, rel = ece([r["conf"] for r in records], [r["correct"] for r in records])
    out = {"n": n, "acc": acc, "nll": sum(r["nll"] for r in records) / n,
           "brier": sum(r["brier"] for r in records) / n, "ece": e,
           "mean_conf": sum(r["conf"] for r in records) / n,
           "chance": sum(1 / r["n_options"] for r in records) / n}
    hard = [r for r in records if not r["blind_easy"]]
    if len(hard) < n:
        out["acc_blind_hard"] = sum(r["correct"] for r in hard) / max(1, len(hard))
        out["n_blind_hard"] = len(hard)
    ords = [r for r in records if "rps" in r]
    if ords:
        out["rps"] = sum(r["rps"] for r in ords) / len(ords)
        out["within1"] = sum(abs(r["pred_level"] - r["true_level"]) <= 1 for r in ords) / len(ords)
        out["mae_expected_level"] = sum(abs(r["exp_level"] - r["true_level"]) for r in ords) / len(ords)
    out["_reliability"] = rel
    return out


def utility_curve(records, costs=((1.0, 0.25), (3.0, 0.25), (10.0, 0.25))):
    """Answer-or-escalate policy: answer iff conf >= tau. Utility per item:
    +1 correct answer, -c_wrong wrong answer, -c_esc escalation. With
    calibrated conf the utility-optimal threshold is known in closed form:
    tau* = (c_wrong - c_esc) / (1 + c_wrong). Reports utility at tau* and the
    best achievable utility over a tau sweep -- the gap measures miscalibration
    in decision terms."""
    out = {}
    for cw, ce in costs:
        tau_star = (cw - ce) / (1 + cw)

        def util(tau):
            u = 0.0
            for r in records:
                if r["conf"] >= tau:
                    u += 1.0 if r["correct"] else -cw
                else:
                    u -= ce
            return u / max(1, len(records))

        best = max((util(t / 50), t / 50) for t in range(51))
        out[f"cw{cw:g}"] = {"tau_star": round(tau_star, 3), "utility_at_tau_star": util(tau_star),
                            "best_utility": best[0], "best_tau": best[1],
                            "answer_rate_at_tau_star": sum(r["conf"] >= tau_star for r in records) / max(1, len(records))}
    return out


def fit_temperatures(records_by_bucket, grid=None):
    """Post-hoc temperature per (qtype, cardinality bucket), NLL-optimal on val.
    records must carry 'logits' (list) and 'y'."""
    grid = grid or [round(0.05 * 1.15 ** i, 4) for i in range(40)]
    out = {}
    for key, recs in records_by_bucket.items():
        if len(recs) < 30:
            continue
        best = None
        for T in grid:
            nll = 0.0
            for r in recs:
                z = [l / T for l in r["logits"]]
                m = max(z)
                lse = m + math.log(sum(math.exp(v - m) for v in z))
                nll += lse - z[r["y"]]
            nll /= len(recs)
            if best is None or nll < best[0]:
                best = (nll, T)
        out[key] = {"T": best[1], "nll": best[0], "n": len(recs)}
    return out


def card_bucket(n):
    return "2" if n <= 2 else "3-5" if n <= 5 else "6-10" if n <= 10 else "11-30" if n <= 30 else "31+"
