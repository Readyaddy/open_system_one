"""Experiment 8 training: System-1 decision model, RLCD-style objective,
continued from exp7a on the rebuilt shortcut-audited corpora.

  python train.py --run_name exp8a --estimator exact              # recommended
  python train.py --run_name exp8a --benchmark 40                 # timing only, no saves

Everything logged lands in <out_dir>/:
  train.log            console mirror
  metrics.jsonl        one line per log interval (train) and per eval (val)
  eval_<step>.json     full eval payload incl. reliability bins + fitted temperatures
  latest.pt / best.pt  checkpoints (latest has optimizer state; best is model only)
and in W&B (project --wandb_project), which also records system metrics.

Logged every --log_every steps:
  train/loss, train/reward (+ log_score, spherical, rps), per-family train acc/reward/count,
  grad_norm, lr/backbone_top, lr/head, inject_gate,
  perf/step_s, perf/examples_s, perf/tokens_s, perf/data_wait_frac, perf/micro_batches,
  gpu/util, gpu/mem_used_gb, gpu/temp_c, gpu/power_w, gpu/sm_clock_mhz (NVML, 2 s sampling),
  cuda/alloc_gb, cuda/peak_gb, cuda/reserved_gb
Every --eval_every steps (and at step 0 = exp7a baseline): the full s1_eval suite.
"""
import argparse
import json
import math
import os
import random
import sys
import threading
import time
from collections import defaultdict

import torch
from torch.amp import autocast

sys.path.insert(0, os.path.dirname(__file__))
from s1_model import S1DecisionModel, PackedSequenceBuilder, get_tokenizer, BACKBONE  # noqa: E402
import s1_data as SD  # noqa: E402
import s1_eval as SE  # noqa: E402
import rlcd  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


# --------------------------------------------------------------------------
# GPU monitor (NVML, background thread)
# --------------------------------------------------------------------------

class GPUMonitor:
    def __init__(self, interval=2.0):
        self.samples, self.interval, self.ok = [], interval, False
        try:
            import pynvml
            pynvml.nvmlInit()
            self.nv, self.h = pynvml, pynvml.nvmlDeviceGetHandleByIndex(0)
            self.ok = True
        except Exception as e:
            print(f"GPU monitor disabled: {e}", flush=True)
        self._stop = threading.Event()
        self.lock = threading.Lock()

    def start(self):
        if self.ok:
            threading.Thread(target=self._run, daemon=True).start()
        return self

    def _run(self):
        nv, h = self.nv, self.h
        while not self._stop.is_set():
            try:
                u = nv.nvmlDeviceGetUtilizationRates(h)
                m = nv.nvmlDeviceGetMemoryInfo(h)
                s = {"util": u.gpu, "mem_util": u.memory, "mem_used_gb": m.used / 2**30,
                     "temp_c": nv.nvmlDeviceGetTemperature(h, 0),
                     "power_w": nv.nvmlDeviceGetPowerUsage(h) / 1000,
                     "sm_clock_mhz": nv.nvmlDeviceGetClockInfo(h, 1)}
                with self.lock:
                    self.samples.append(s)
            except Exception:
                pass
            time.sleep(self.interval)

    def drain(self):
        with self.lock:
            xs, self.samples = self.samples, []
        if not xs:
            return {}
        out = {f"gpu/{k}": sum(x[k] for x in xs) / len(xs) for k in xs[0]}
        out["gpu/util_min"] = min(x["util"] for x in xs)
        return out


# --------------------------------------------------------------------------

def param_groups(model, base_lr, head_mult, decay, wd):
    groups = []
    layers = list(model.backbone.layers)
    n = len(layers)
    for i, layer in enumerate(layers):
        groups.append({"params": [p for p in layer.parameters() if p.requires_grad],
                       "lr": base_lr * decay ** (n - 1 - i), "name": f"layer{i}", "weight_decay": wd})
    in_layers = {id(p) for l in layers for p in l.parameters()}
    bb_other = [p for p in model.backbone.parameters() if id(p) not in in_layers and p.requires_grad]
    groups.append({"params": bb_other, "lr": base_lr * decay ** n, "name": "backbone_other", "weight_decay": wd})
    bb = {id(p) for p in model.backbone.parameters()}
    head = [p for p in model.parameters() if id(p) not in bb and p.requires_grad]
    groups.append({"params": head, "lr": base_lr * head_mult, "name": "head", "weight_decay": wd})
    for g in groups:
        g["base_lr"] = g["lr"]
    return [g for g in groups if g["params"]]


class Log:
    def __init__(self, out_dir, use_wandb):
        self.f = open(os.path.join(out_dir, "train.log"), "a", encoding="utf-8")
        self.jsonl = os.path.join(out_dir, "metrics.jsonl")
        self.use_wandb = use_wandb

    def p(self, *a):
        s = " ".join(str(x) for x in a)
        print(s, flush=True)
        self.f.write(s + "\n")
        self.f.flush()

    def metrics(self, step, d, kind):
        clean = {k: (float(v) if isinstance(v, (int, float)) else v) for k, v in d.items()
                 if isinstance(v, (int, float))}
        with open(self.jsonl, "a") as f:
            f.write(json.dumps({"step": step, "kind": kind, "time": time.time(), **clean}) + "\n")
        if self.use_wandb:
            import wandb
            wandb.log(clean, step=step)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_name", default="exp8a")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--init_ckpt", default=os.path.join(ROOT, "exp7_best_zeroshot.pt"))
    ap.add_argument("--resume", action="store_true", help="continue from <out_dir>/latest.pt")
    ap.add_argument("--head_input", default="s0")
    # objective
    ap.add_argument("--estimator", choices=["exact", "grpo", "hybrid"], default="exact")
    ap.add_argument("--grpo_G", type=int, default=8)
    ap.add_argument("--grpo_kappa", type=float, default=50.0)
    ap.add_argument("--hybrid_lambda", type=float, default=0.5)
    # optimisation
    ap.add_argument("--max_steps", type=int, default=6000)
    ap.add_argument("--examples_per_step", type=int, default=32)
    ap.add_argument("--max_tokens", type=int, default=8000, help="per micro-batch, padded tokens")
    ap.add_argument("--max_options", type=int, default=256,
                    help="option strings per micro-batch; each is a separate backbone pass WITH grads")
    ap.add_argument("--save_every", type=int, default=250)
    ap.add_argument("--baseline_json", default=None,
                    help="log an existing step-0 eval JSON instead of re-running the baseline eval")
    ap.add_argument("--gpu_mem_fraction", type=float, default=0.88,
                    help="Cap on PyTorch's CUDA memory. Without it the caching allocator grew to the "
                         "full 12 GB card by step 60 of exp8a and the Windows driver silently spilled "
                         "into shared system RAM: step time 3.9s -> 8s, power 93W -> 80W, clocks UP "
                         "(so not thermal). With a cap the allocator frees cached blocks and retries "
                         "instead (counted in cuda/alloc_retries).")
    ap.add_argument("--base_lr", type=float, default=1e-5, help="top backbone layer")
    ap.add_argument("--layer_decay", type=float, default=0.9)
    ap.add_argument("--head_mult", type=float, default=5.0)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_frac", type=float, default=0.03)
    ap.add_argument("--min_lr_frac", type=float, default=0.1)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--no_grad_ckpt", action="store_true")
    # data mix
    for k, v in SD.DataCfg().__dict__.items():
        ap.add_argument(f"--{k}", type=float, default=v)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=8)
    # eval / logging
    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--val_per_source", type=int, default=300)
    ap.add_argument("--val_intent", type=int, default=1000)
    ap.add_argument("--skip_baseline_eval", action="store_true")
    ap.add_argument("--with_bool_desc", action="store_true",
                    help="add bool/<src>@desc and typed/bool@swapped val groups (exp8b)")
    ap.add_argument("--patience", type=int, default=5, help="evals without select/score improvement")
    ap.add_argument("--log_every", type=int, default=20)
    ap.add_argument("--wandb_project", default="open-system-one")
    ap.add_argument("--no_wandb", action="store_true")
    ap.add_argument("--benchmark", type=int, default=0, help="run N steps for timing, then exit")
    args = ap.parse_args()

    if os.name == "nt":
        # Keep the laptop awake while training. exp8a3 hit a Modern Standby
        # "idle timeout" at step 1500 (GPU fell to 36 W, a 4-min eval took
        # 25 min) and an unclean reboot after waking. ES_SYSTEM_REQUIRED alone
        # did NOT stop it recurring at step 3000: Modern Standby starts when
        # the display turns off, so ES_DISPLAY_REQUIRED is needed too. Held
        # only while this process runs -- no power settings are changed.
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001 | 0x00000002)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    dev = torch.device("cuda")
    torch.cuda.set_per_process_memory_fraction(args.gpu_mem_fraction, 0)
    out_dir = args.out_dir or os.path.join(ROOT, "checkpoints", "exp8_s1_rlcd", args.run_name)
    os.makedirs(out_dir, exist_ok=True)
    use_wandb = not args.no_wandb and not args.benchmark
    L = Log(out_dir, use_wandb)
    L.p(f"\n===== exp8 {args.run_name} | {time.strftime('%Y-%m-%d %H:%M:%S')} =====")
    L.p("args: " + json.dumps(vars(args)))

    cfg = SD.DataCfg(**{k: getattr(args, k) for k in SD.DataCfg().__dict__})
    tok = get_tokenizer(BACKBONE)
    builder = PackedSequenceBuilder(tok, budget_total=2048, l_context=768, l_instructions=96, l_max_per_option=64)
    packer = SD.Packer(builder, tok, max_tokens=args.max_tokens, max_options=args.max_options)
    # JevBench groups are scored with the official protocol at full context; training stays at 768.
    long_packer = SE.long_context_packer(tok)

    t0 = time.time()
    C = SD.load_corpora()
    mix = SD.TrainSampler(C, cfg, seed=0).mix_table()
    L.p(f"corpora loaded in {time.time()-t0:.0f}s. effective sampling mix:")
    for k, v in sorted(mix.items(), key=lambda x: -x[1]):
        L.p(f"    {k:34s} {v*100:5.2f}%")
    suite = SD.build_val_suite(C, cfg, {"per_source": args.val_per_source, "intent": args.val_intent},
                               with_bool_desc=args.with_bool_desc)
    L.p(f"val suite: {len(suite)} groups, {sum(len(v) for v in suite.values())} examples")

    model = S1DecisionModel(mask_token_id=tok.mask_token_id, head_input=args.head_input,
                            gradient_checkpointing=not args.no_grad_ckpt)
    start_step, best, bad = 0, -1.0, 0
    latest = os.path.join(out_dir, "latest.pt")
    resume_ck = None
    if args.resume and os.path.exists(latest):
        resume_ck = torch.load(latest, map_location="cpu", weights_only=False)
        model.load_state_dict(resume_ck["model_state"], strict=True)
        start_step, best, bad = resume_ck["step"], resume_ck["best"], resume_ck["bad"]
        L.p(f"resumed from {latest} at step {start_step} (best {best:.4f})")
    else:
        ck, miss, unexp = model.load_exp7a(args.init_ckpt, strict=True)
        where = f"epoch {ck['epoch']}" if "epoch" in ck else f"step {ck.get('step')}"
        L.p(f"initialized from {args.init_ckpt} ({where}, strict load OK)")
        del ck
    model.to(dev)
    n_all = sum(p.numel() for p in model.parameters())
    L.p(f"params: {n_all/1e6:.1f}M total, head {sum(p.numel() for n,p in model.named_parameters() if not n.startswith('backbone.'))/1e6:.1f}M")

    groups = param_groups(model, args.base_lr, args.head_mult, args.layer_decay, args.weight_decay)
    try:
        import bitsandbytes as bnb
        opt = bnb.optim.AdamW8bit(groups, betas=(0.9, 0.98))
        L.p("optimizer: bitsandbytes AdamW8bit")
    except Exception:
        opt = torch.optim.AdamW(groups, betas=(0.9, 0.98))
        L.p("optimizer: torch AdamW")
    if resume_ck is not None:
        opt.load_state_dict(resume_ck["optimizer_state"])
        del resume_ck

    warm = max(1, int(args.warmup_frac * args.max_steps))

    def lr_factor(step):
        if step < warm:
            return (step + 1) / warm
        prog = (step - warm) / max(1, args.max_steps - warm)
        return args.min_lr_frac + (1 - args.min_lr_frac) * 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))

    if use_wandb:
        import wandb
        wandb.init(project=args.wandb_project, name=args.run_name, id=f"{args.run_name}", resume="allow",
                   config={**vars(args), "sampling_mix": mix}, dir=out_dir)

    def evaluate(step):
        flat, side = SE.run_suite(model, packer, suite, dev, long_packer=long_packer)
        SE.print_report(flat, f"eval @ step {step}")
        with open(os.path.join(out_dir, f"eval_{step:06d}.json"), "w") as f:
            json.dump({"step": step, "metrics": flat, **side}, f, indent=1)
        L.f.write(json.dumps({"eval_step": step, **{k: v for k, v in flat.items() if isinstance(v, (int, float))}}) + "\n")
        L.metrics(step, flat, "eval")
        model.train()
        return flat

    if args.baseline_json and start_step == 0:
        with open(args.baseline_json) as f:
            L.metrics(0, json.load(f)["metrics"], "eval")
        L.p(f"baseline eval logged from {args.baseline_json}")
    elif args.benchmark == 0 and start_step == 0 and not args.skip_baseline_eval:
        L.p("\n--- baseline eval: exp7a as loaded, before any exp8 training ---")
        evaluate(0)

    stream = SD.StepStream(C, cfg, builder, tok, args.examples_per_step, args.max_tokens,
                           seed=args.seed * 7919 + start_step, max_options=args.max_options)
    loader = torch.utils.data.DataLoader(stream, batch_size=None, num_workers=args.num_workers,
                                         pin_memory=True, prefetch_factor=4 if args.num_workers else None,
                                         persistent_workers=args.num_workers > 0)
    it = iter(loader)
    gpu = GPUMonitor().start()
    model.train()
    fam_names = ["intent", "mcq", "bool", "score", "diversity"]

    acc_int = defaultdict(float)
    t_int, wait_int, steps_int, tok_int, ex_int, mb_int = time.time(), 0.0, 0, 0, 0, 0
    oom_total, oom_streak = 0, 0
    step = start_step
    t_start = time.time()
    torch.cuda.reset_peak_memory_stats()

    def run_micro_batches(mbs, n_ex):
        """Forward + backward over one step's micro-batches, accumulating
        per-example stats on the GPU (no host sync until the next log line)."""
        stats = defaultdict(float)
        for mb in mbs:
            packed, o = mb.to(dev)
            with autocast("cuda", dtype=torch.bfloat16):
                logits = model(packed, o)
            loss, R, comps, _ = rlcd.objective(
                logits.float(), packed.answer_idx, packed.qtype_idx, packed.valid_mask,
                estimator=args.estimator, lam=args.hybrid_lambda,
                **({} if args.estimator == "exact" else {"G": args.grpo_G, "kappa": args.grpo_kappa}))
            (loss * len(mb.metas) / n_ex).backward()
            with torch.no_grad():
                correct = (logits.argmax(-1) == packed.answer_idx).float()
                stats["loss"] += loss.detach() * len(mb.metas)
                stats["reward"] += R.sum()
                stats["log_score"] += comps["log_score"].sum()
                stats["spherical"] += comps["spherical"].sum()
                stats["rps_sum"] += (comps["rps"] * comps["is_ordinal"]).sum()
                stats["rps_n"] += comps["is_ordinal"].sum()
                stats["correct"] += correct.sum()
                fam_idx = torch.tensor([fam_names.index(m.family) for m in mb.metas], device=dev)
                for fi, k in enumerate(fam_names):
                    sel = (fam_idx == fi).float()
                    stats[f"n_{k}"] += sel.sum()
                    stats[f"c_{k}"] += (correct * sel).sum()
                    stats[f"r_{k}"] += (R * sel).sum()
        return stats

    def save_latest(step):
        torch.save({"model_state": model.state_dict(), "optimizer_state": opt.state_dict(), "step": step,
                    "best": best, "bad": bad, "args": vars(args)}, latest)

    while step < args.max_steps:
        tw = time.time()
        mbs = next(it)
        wait_int += time.time() - tw
        f = lr_factor(step)
        for g in opt.param_groups:
            g["lr"] = g["base_lr"] * f
        n_ex = sum(len(mb.metas) for mb in mbs)

        try:
            stats = run_micro_batches(mbs, n_ex)
        except torch.OutOfMemoryError:
            # A rare oversized batch: drop this step instead of the whole run.
            del mbs
            opt.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            oom_total += 1
            oom_streak += 1
            L.p(f"  !! OOM at step {step}: step skipped ({oom_total} total, {oom_streak} in a row)")
            if oom_streak >= 5:
                raise RuntimeError("5 consecutive OOM skips -- lower --max_tokens / --max_options")
            continue
        oom_streak = 0
        for k, v in stats.items():
            acc_int[k] += v
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        acc_int["grad_norm"] += gn.detach()
        opt.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        steps_int += 1
        ex_int += n_ex
        tok_int += sum(mb.n_tokens for mb in mbs)
        mb_int += len(mbs)

        if step % args.log_every == 0 or step == args.max_steps or (args.benchmark and step - start_step == args.benchmark):
            dt = time.time() - t_int
            a = {k: (v.item() if torch.is_tensor(v) else v) for k, v in acc_int.items()}
            d = {"train/loss": a["loss"] / ex_int, "train/reward": a["reward"] / ex_int,
                 "train/log_score": a["log_score"] / ex_int, "train/spherical": a["spherical"] / ex_int,
                 "train/rps": a["rps_sum"] / max(1, a["rps_n"]), "train/acc": a["correct"] / ex_int,
                 "train/oom_skips": oom_total,
                 "grad_norm": a["grad_norm"] / steps_int, "inject_gate": model.inject_gate.item(),
                 "lr/backbone_top": args.base_lr * f, "lr/head": args.base_lr * args.head_mult * f,
                 "perf/step_s": dt / steps_int, "perf/examples_s": ex_int / dt, "perf/tokens_s": tok_int / dt,
                 "perf/data_wait_frac": wait_int / dt, "perf/micro_batches": mb_int / steps_int,
                 "cuda/alloc_gb": torch.cuda.memory_allocated() / 2**30,
                 "cuda/peak_gb": torch.cuda.max_memory_allocated() / 2**30,
                 "cuda/reserved_gb": torch.cuda.memory_reserved() / 2**30,
                 "cuda/alloc_retries": torch.cuda.memory_stats().get("num_alloc_retries", 0), **gpu.drain()}
            for k in fam_names:
                n = a.get(f"n_{k}", 0)
                if n:
                    d[f"train_fam/{k}/acc"] = a[f"c_{k}"] / n
                    d[f"train_fam/{k}/reward"] = a[f"r_{k}"] / n
                    d[f"train_fam/{k}/frac"] = n / ex_int
            eta = (args.max_steps - step) * (time.time() - t_start) / max(1, step - start_step)
            L.p(f"  step {step:6d}/{args.max_steps} loss {d['train/loss']:.4f} R {d['train/reward']:+.4f} "
                f"acc {d['train/acc']*100:5.1f} | " +
                " ".join(f"{k[:3]} {d.get(f'train_fam/{k}/acc', 0)*100:4.1f}" for k in fam_names) +
                f" | gn {d['grad_norm']:.2f} lr {d['lr/backbone_top']:.1e} gate {d['inject_gate']:.4f} | "
                f"{d['perf/step_s']:.2f}s/step {d['perf/examples_s']:.1f}ex/s wait {d['perf/data_wait_frac']*100:.0f}% | "
                f"gpu {d.get('gpu/util', 0):.0f}% {d.get('gpu/mem_used_gb', 0):.1f}GB {d.get('gpu/temp_c', 0):.0f}C "
                f"{d.get('gpu/power_w', 0):.0f}W peak {d['cuda/peak_gb']:.1f}GB rsv {d['cuda/reserved_gb']:.1f}GB "
                f"retries {d['cuda/alloc_retries']} oom {oom_total} | ETA {eta/3600:.1f}h")
            L.metrics(step, d, "train")
            acc_int.clear()
            t_int, wait_int, steps_int, tok_int, ex_int, mb_int = time.time(), 0.0, 0, 0, 0, 0
            torch.cuda.reset_peak_memory_stats()

        if args.benchmark and step - start_step >= args.benchmark:
            L.p(f"benchmark done: {args.benchmark} steps")
            return

        if step % args.eval_every == 0 or step == args.max_steps:
            flat = evaluate(step)
            score = flat["select/score"]
            if score > best:
                best, bad = score, 0
                torch.save({"model_state": model.state_dict(), "step": step, "select_score": score,
                            "args": vars(args)}, os.path.join(out_dir, "best.pt"))
                L.p(f"  *** new best select/score {score*100:.2f} -> best.pt")
            else:
                bad += 1
                L.p(f"  no improvement ({bad}/{args.patience}); best {best*100:.2f}")
            save_latest(step)
            if bad >= args.patience:
                L.p("early stop")
                break
        elif step % args.save_every == 0:
            save_latest(step)
            L.p(f"  saved latest.pt @ step {step}")
    L.p("done.")
    if use_wandb:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
