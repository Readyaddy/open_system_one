"""Experiment 9 training: the exp7/8 decision architecture on a ~4B embedding
LLM (Qwen3-Embedding-4B), same data / RLCD loss / metrics as exp8.

  local check : python s9_train.py --backbone Qwen/Qwen3-Embedding-0.6B --run_name exp9_local --max_steps 60 ...
  Colab       : see scripts/colab_exp9_*.sh (smoke, then budgeted run)

Budget safety (Colab credits are the constraint):
  --max_hours H     stop training at H hours minus the time the final eval is
                    expected to take, run the final eval, save, exit cleanly.
  --release_vm      after the final save, release the Colab VM from inside
                    (google.colab.runtime.unassign) so it cannot idle and burn units.
Checkpoints hold only what trains (LoRA + head, ~0.4B params), never the frozen base.
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
import torch.nn.functional as F
from torch.amp import autocast

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "exp8_s1_rlcd"))
import s1_data as SD  # noqa: E402
import s1_eval as SE  # noqa: E402
import rlcd  # noqa: E402
import s9_eval as E9  # noqa: E402  (also patches SE.predict -> predict9)
from s9_model import S9DecisionModel, Packer9, get_tokenizer  # noqa: E402

FAM = ["intent", "mcq", "bool", "score", "diversity"]


class GPUMonitor:
    def __init__(self, interval=2.0):
        self.samples, self.interval, self.ok, self.lock = [], interval, False, threading.Lock()
        try:
            import pynvml
            pynvml.nvmlInit()
            self.nv, self.h, self.ok = pynvml, pynvml.nvmlDeviceGetHandleByIndex(0), True
        except Exception as e:
            print(f"GPU monitor disabled: {e}", flush=True)

    def start(self):
        if self.ok:
            threading.Thread(target=self._run, daemon=True).start()
        return self

    def _run(self):
        nv, h = self.nv, self.h
        while True:
            try:
                u, m = nv.nvmlDeviceGetUtilizationRates(h), nv.nvmlDeviceGetMemoryInfo(h)
                s = {"util": u.gpu, "mem_used_gb": m.used / 2**30, "temp_c": nv.nvmlDeviceGetTemperature(h, 0),
                     "power_w": nv.nvmlDeviceGetPowerUsage(h) / 1000}
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


class StepStream9(torch.utils.data.IterableDataset):
    def __init__(self, C, cfg, packer, examples_per_step, seed):
        self.C, self.cfg, self.packer, self.eps, self.seed = C, cfg, packer, examples_per_step, seed

    def __iter__(self):
        wi = torch.utils.data.get_worker_info()
        sampler = SD.TrainSampler(self.C, self.cfg, seed=self.seed * 1000 + (wi.id if wi else 0))
        while True:
            yield self.packer.pack([sampler.sample() for _ in range(self.eps)])


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
        clean = {k: float(v) for k, v in d.items() if isinstance(v, (int, float))}
        with open(self.jsonl, "a") as f:
            f.write(json.dumps({"step": step, "kind": kind, "time": time.time(), **clean}) + "\n")
        if self.use_wandb:
            import wandb
            wandb.log(clean, step=step)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_name", default="exp9a")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--backbone", default="Qwen/Qwen3-Embedding-4B")
    ap.add_argument("--full_finetune", action="store_true", help="train the whole backbone instead of LoRA")
    ap.add_argument("--lora_r", type=int, default=64)
    ap.add_argument("--lora_alpha", type=int, default=128)
    ap.add_argument("--n_codes", type=int, default=16)
    ap.add_argument("--gate_init", type=float, default=0.1)
    # scaled decision head (s9_head.py)
    ap.add_argument("--head", choices=["rethink", "simple"], default="rethink")
    ap.add_argument("--attn", choices=["sdpa", "flex_attention"], default="sdpa",
                    help="flex_attention needs Triton (Linux); verify with test_flex.py first")
    ap.add_argument("--k_max", type=int, default=6, help="rethinking passes")
    ap.add_argument("--n_latents", type=int, default=128, help="resampler latents (context pooling)")
    ap.add_argument("--resampler_layers", type=int, default=2)
    ap.add_argument("--block_layers", type=int, default=4, help="decoder layers in the recurrent block")
    ap.add_argument("--n_scratch", type=int, default=8, help="scratchpad slots per pass (0 = off)")
    ap.add_argument("--depth_weighting", choices=["uniform", "ascending"], default="uniform")
    ap.add_argument("--monotonic_weight", type=float, default=0.0)
    ap.add_argument("--resume", action="store_true")
    # objective (same as exp8)
    ap.add_argument("--estimator", choices=["exact", "grpo", "hybrid"], default="exact")
    # optimisation
    ap.add_argument("--max_steps", type=int, default=4000)
    ap.add_argument("--max_hours", type=float, default=0.0, help="wall-clock budget incl. final eval (0 = off)")
    ap.add_argument("--examples_per_step", type=int, default=32)
    ap.add_argument("--max_tokens", type=int, default=16000, help="padded tokens per micro-batch")
    ap.add_argument("--max_options", type=int, default=512)
    ap.add_argument("--lr_lora", type=float, default=1e-4)
    ap.add_argument("--lr_backbone", type=float, default=1e-5, help="only with --full_finetune")
    ap.add_argument("--lr_head", type=float, default=1e-4)
    ap.add_argument("--lr_gate", type=float, default=3e-3)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_frac", type=float, default=0.03)
    ap.add_argument("--min_lr_frac", type=float, default=0.1)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--no_grad_ckpt", action="store_true")
    # packing
    ap.add_argument("--budget_total", type=int, default=2048)
    ap.add_argument("--l_context", type=int, default=768)
    # data mix (DataCfg fields) -- exp8b's mix incl. descriptive bool
    for k, v in SD.DataCfg().__dict__.items():
        ap.add_argument(f"--{k}", type=float, default=v)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=9)
    # eval / logging
    ap.add_argument("--eval_every", type=int, default=1000,
                    help="a full eval on 4B costs ~10-15 min of GPU; every 1000 steps keeps it under ~15% of the budget")
    ap.add_argument("--val_per_source", type=int, default=300)
    ap.add_argument("--val_intent", type=int, default=1000)
    ap.add_argument("--val_typed", type=int, default=500)
    ap.add_argument("--save_every", type=int, default=250)
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--wandb_project", default="open-system-one")
    ap.add_argument("--no_wandb", action="store_true")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny val suite + frequent eval/save + a resume round-trip; for pre-flight checks")
    ap.add_argument("--release_vm", action="store_true")
    ap.add_argument("--sync_dir", default=None,
                    help="Drive folder to mirror results into. Training writes to --out_dir on LOCAL disk; "
                         "the Colab Drive FUSE mount dropped mid-transfer once (Errno 107), so it is only ever "
                         "a best-effort, retried copy target -- never something training depends on.")
    args = ap.parse_args()
    if not any(a.startswith("--p_bool_desc") for a in sys.argv):
        args.p_bool_desc = 0.5

    t_begin = time.time()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    dev = torch.device("cuda")
    root = os.path.abspath(os.path.join(HERE, "..", ".."))
    out_dir = args.out_dir or os.path.join(root, "checkpoints", "exp9_scale", args.run_name)
    os.makedirs(out_dir, exist_ok=True)
    use_wandb = not args.no_wandb
    L = Log(out_dir, use_wandb)
    L.p(f"\n===== exp9 {args.run_name} | {time.strftime('%Y-%m-%d %H:%M:%S')} | {torch.cuda.get_device_name(0)} =====")
    L.p("args: " + json.dumps(vars(args)))

    cfg = SD.DataCfg(**{k: getattr(args, k) for k in SD.DataCfg().__dict__})
    tok = get_tokenizer(args.backbone)
    packer = Packer9(tok, budget_total=args.budget_total, l_context=args.l_context,
                     max_tokens=args.max_tokens, max_options=args.max_options)
    t0 = time.time()
    C = SD.load_corpora()
    L.p(f"corpora loaded in {time.time() - t0:.0f}s")
    sizes = {"per_source": 40 if args.smoke else args.val_per_source, "intent": 100 if args.smoke else args.val_intent,
             "typed": 60 if args.smoke else args.val_typed}
    suite = SD.build_val_suite(C, cfg, sizes, with_bool_desc=True)
    sweep = E9.build_sweep(C, cfg, n=40 if args.smoke else 300)
    L.p(f"val suite: {len(suite)} groups / {sum(len(v) for v in suite.values())} ex; sweep {len(sweep)} groups")

    model = S9DecisionModel(args.backbone, lora_r=args.lora_r, lora_alpha=args.lora_alpha,
                            full_finetune=args.full_finetune, n_codes=args.n_codes, gate_init=args.gate_init,
                            gradient_checkpointing=not args.no_grad_ckpt, head=args.head, attn_impl=args.attn,
                            n_latents=args.n_latents, resampler_layers=args.resampler_layers,
                            block_layers=args.block_layers, k_max=args.k_max, n_scratch=args.n_scratch)
    model.to(dev)
    pc = model.param_counts()
    L.p(f"params: total {pc['total']/1e9:.3f}B | trainable {pc['trainable']/1e6:.1f}M | frozen backbone {pc['frozen_backbone']/1e9:.3f}B")

    groups, seen = [], set()

    def add(name, params, lr, wd):
        params = [p for p in params if p.requires_grad and id(p) not in seen]
        seen.update(id(p) for p in params)
        if params:
            groups.append({"name": name, "params": params, "lr": lr, "base_lr": lr, "weight_decay": wd})

    add("gate", [model.inject_gate], args.lr_gate, 0.0)
    bb_lr = args.lr_backbone if args.full_finetune else args.lr_lora
    add("backbone", [p for n, p in model.named_parameters() if n.startswith("backbone.")], bb_lr, args.weight_decay)
    add("head", list(model.parameters()), args.lr_head, args.weight_decay)
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.98), fused=True)
    L.p("param groups: " + ", ".join(f"{g['name']} {sum(p.numel() for p in g['params'])/1e6:.1f}M @ {g['lr']:.0e}" for g in groups))

    start, best, bad = 0, -1.0, 0
    latest = os.path.join(out_dir, "latest.pt")
    drive_latest = os.path.join(args.sync_dir, "latest_model_bf16.pt") if args.sync_dir else None
    if args.resume and os.path.exists(latest):
        ck = torch.load(latest, map_location="cpu", weights_only=False)
        model.load_trainable_state_dict(ck["model_state"])
        opt.load_state_dict(ck["optimizer_state"])
        start, best, bad = ck["step"], ck["best"], ck["bad"]
        L.p(f"resumed from {latest} at step {start} (best {best:.4f})")
        del ck
    elif args.resume and drive_latest and os.path.exists(drive_latest):
        # New VM: the full optimizer-state checkpoint lived on the old VM's local disk.
        # Resume the weights from the Drive mirror; the optimizer restarts (warmup re-applies).
        ck = torch.load(drive_latest, map_location="cpu", weights_only=False)
        model.load_trainable_state_dict(ck["model_state"])
        start, best, bad = ck["step"], ck["best"], ck["bad"]
        L.p(f"resumed WEIGHTS from Drive mirror {drive_latest} at step {start} (best {best:.4f}); fresh optimizer")
        del ck

    warm = max(1, int(args.warmup_frac * args.max_steps))

    def lr_factor(s):
        if s < warm:
            return (s + 1) / warm
        prog = min(1.0, (s - warm) / max(1, args.max_steps - warm))
        return args.min_lr_frac + (1 - args.min_lr_frac) * 0.5 * (1 + math.cos(math.pi * prog))

    if use_wandb:
        import wandb
        wandb.init(project=args.wandb_project, name=args.run_name, id=args.run_name, resume="allow",
                   config={**vars(args), **{f"params/{k}": v for k, v in pc.items()}}, dir=out_dir)

    def save(path, step, with_opt):
        tmp = path + ".tmp"
        torch.save({"model_state": model.trainable_state_dict(), "step": step, "best": best, "bad": bad,
                    "args": vars(args), **({"optimizer_state": opt.state_dict()} if with_opt else {})}, tmp)
        os.replace(tmp, path)

    def save_bf16(path, step):
        """Compact model-only copy (~1.8 GB) for the Drive mirror; the full latest.pt
        with optimizer state (~10 GB) stays on local disk."""
        tmp = path + ".tmp"
        sd = {k: (v.to(torch.bfloat16) if v.is_floating_point() else v) for k, v in model.trainable_state_dict().items()}
        torch.save({"model_state": sd, "step": step, "best": best, "bad": bad, "args": vars(args)}, tmp)
        os.replace(tmp, path)

    def sync(with_models):
        """Mirror results to Drive with retries. Never raises: a failed copy is logged
        and reported, training carries on from local disk."""
        if not args.sync_dir:
            return True
        import shutil
        names = [f for f in os.listdir(out_dir)
                 if f.endswith((".json", ".jsonl", ".log")) or f == "DONE"]
        if with_models:
            names += [f for f in ("best_bf16.pt", "latest_model_bf16.pt") if os.path.exists(os.path.join(out_dir, f))]
        ok = True
        for name in names:
            src, dst = os.path.join(out_dir, name), os.path.join(args.sync_dir, name)
            for attempt in range(4):
                try:
                    os.makedirs(args.sync_dir, exist_ok=True)
                    shutil.copyfile(src, dst + ".tmp")
                    os.replace(dst + ".tmp", dst)
                    if os.path.getsize(dst) != os.path.getsize(src):
                        raise IOError("size mismatch after copy")
                    break
                except Exception as e:
                    L.p(f"  sync {name} failed (attempt {attempt + 1}/4): {e}")
                    time.sleep(15 * (attempt + 1))
            else:
                ok = False
        return ok

    eval_seconds = [0.0]

    def evaluate(step):
        te = time.time()
        flat, side = E9.evaluate_all(model, packer, suite, sweep, dev, tok, with_order=True)
        eval_seconds[0] = time.time() - te
        flat["eval/seconds"] = eval_seconds[0]
        SE.print_report(flat, f"eval @ step {step}")
        E9.print_sweep(flat)
        E9.print_depth(flat)
        with open(os.path.join(out_dir, f"eval_{step:06d}.json"), "w") as f:
            json.dump({"step": step, "metrics": flat, **side}, f, indent=1)
        L.metrics(step, flat, "eval")
        model.train()
        return flat

    loader = torch.utils.data.DataLoader(StepStream9(C, cfg, packer, args.examples_per_step, args.seed * 7919 + start),
                                         batch_size=None, num_workers=args.num_workers, pin_memory=True,
                                         prefetch_factor=4 if args.num_workers else None,
                                         persistent_workers=args.num_workers > 0)
    it = iter(loader)
    gpu = GPUMonitor().start()
    model.train()
    acc = defaultdict(float)
    t_int, wait, n_steps, n_ex, n_tok, n_mb = time.time(), 0.0, 0, 0, 0, 0
    oom_total, oom_streak = 0, 0
    step = start
    t_train0 = time.time()
    torch.cuda.reset_peak_memory_stats()
    stop_reason = "max_steps"

    while step < args.max_steps:
        if args.max_hours > 0:
            reserve = max(eval_seconds[0], 600) * 1.3 + 300   # final eval + save
            if time.time() - t_begin > args.max_hours * 3600 - reserve:
                stop_reason = f"time budget ({args.max_hours}h)"
                break
        tw = time.time()
        mbs = next(it)
        wait += time.time() - tw
        f = lr_factor(step)
        for g in opt.param_groups:
            g["lr"] = g["base_lr"] * f
        ex_step = sum(len(m) for _, m in mbs)
        try:
            stats = defaultdict(float)
            for batch, metas in mbs:
                b = batch.to(dev)
                with autocast("cuda", dtype=torch.bfloat16):
                    outs = model(b, all_depths=True)
                logits = outs[-1]
                K = len(outs)
                w = torch.arange(1, K + 1, dtype=torch.float32) if args.depth_weighting == "ascending" else torch.ones(K)
                w = (w / w.sum()).tolist()
                loss = 0.0
                for kk, o in enumerate(outs):
                    lk, Rk, ck, _ = rlcd.objective(o, b.answer_idx, b.qtype_idx, b.valid_mask, estimator=args.estimator)
                    loss = loss + w[kk] * lk
                    if kk == K - 1:
                        R, comps = Rk, ck
                if args.monotonic_weight > 0 and K > 1:
                    marg = [o.gather(1, b.answer_idx[:, None]).squeeze(1)
                            - torch.logsumexp(o.scatter(1, b.answer_idx[:, None], float("-inf")), -1) for o in outs]
                    mono = torch.stack([F.relu(marg[i - 1] - marg[i]).mean() for i in range(1, K)]).mean()
                    loss = loss + args.monotonic_weight * mono
                    stats["mono"] += mono.detach() * len(metas)
                (loss * len(metas) / ex_step).backward()
                with torch.no_grad():
                    correct = (logits.argmax(-1) == b.answer_idx).float()
                    stats["loss"] += loss.detach() * len(metas)
                    stats["reward"] += R.sum()
                    stats["rps_sum"] += (comps["rps"] * comps["is_ordinal"]).sum()
                    stats["rps_n"] += comps["is_ordinal"].sum()
                    stats["correct"] += correct.sum()
                    for kk, o in enumerate(outs):
                        stats[f"d{kk + 1}"] += (o.argmax(-1) == b.answer_idx).float().sum()
                    fi = torch.tensor([FAM.index(m.family) for m in metas], device=dev)
                    for j, k in enumerate(FAM):
                        sel = (fi == j).float()
                        stats[f"n_{k}"] += sel.sum()
                        stats[f"c_{k}"] += (correct * sel).sum()
                n_tok += batch.n_tokens
            n_mb += len(mbs)
        except torch.OutOfMemoryError:
            # Drop EVERY reference to the failed micro-batch's graph before freeing.
            # exp9a (2026-09-27) died at step 424 with 5 OOMs in a row: `outs`/`logits`/
            # `loss` are locals of main(), so after the first OOM they still pinned the
            # failed forward's activations and every following batch OOMed too.
            mbs = batch = b = outs = logits = loss = R = comps = stats = None
            lk = Rk = ck = o = marg = mono = correct = None  # noqa: F841
            opt.zero_grad(set_to_none=True)
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            oom_total += 1
            oom_streak += 1
            L.p(f"  !! OOM at step {step}: skipped ({oom_total} total, {oom_streak} in a row)")
            if oom_streak >= 5:
                raise RuntimeError("5 consecutive OOM skips -- lower --max_tokens / --max_options")
            continue
        oom_streak = 0
        for k, v in stats.items():
            acc[k] += v
        gn = torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]], args.clip)
        acc["grad_norm"] += gn.detach()
        opt.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        n_steps += 1
        n_ex += ex_step

        if step % args.log_every == 0 or step == args.max_steps:
            dt = time.time() - t_int
            a = {k: (v.item() if torch.is_tensor(v) else v) for k, v in acc.items()}
            d = {"train/loss": a["loss"] / n_ex, "train/reward": a["reward"] / n_ex, "train/acc": a["correct"] / n_ex,
                 "train/rps": a["rps_sum"] / max(1, a["rps_n"]), "train/oom_skips": oom_total,
                 "grad_norm": a["grad_norm"] / n_steps, "inject_gate": model.inject_gate.item(),
                 "lr/backbone": bb_lr * f, "lr/head": args.lr_head * f,
                 "perf/step_s": dt / n_steps, "perf/examples_s": n_ex / dt, "perf/tokens_s": n_tok / dt,
                 "perf/data_wait_frac": wait / dt, "perf/micro_batches": n_mb / n_steps,
                 "cuda/peak_gb": torch.cuda.max_memory_allocated() / 2**30,
                 "cuda/reserved_gb": torch.cuda.memory_reserved() / 2**30,
                 "time/hours": (time.time() - t_begin) / 3600, **gpu.drain()}
            for k in FAM:
                if a.get(f"n_{k}"):
                    d[f"train_fam/{k}/acc"] = a[f"c_{k}"] / a[f"n_{k}"]
            for kk in range(1, args.k_max + 1):
                if f"d{kk}" in a:
                    d[f"train_depth/{kk}/acc"] = a[f"d{kk}"] / n_ex
            if "mono" in a:
                d["train/monotonic"] = a["mono"] / n_ex
            per_step = (time.time() - t_train0) / max(1, step - start)
            eta = (args.max_steps - step) * per_step
            L.p(f"  step {step:6d}/{args.max_steps} loss {d['train/loss']:.4f} acc {d['train/acc']*100:5.1f} | "
                + " ".join(f"{k[:3]} {d.get(f'train_fam/{k}/acc', 0)*100:4.1f}" for k in FAM)
                + f" | gn {d['grad_norm']:.2f} gate {d['inject_gate']:.4f} | {d['perf/step_s']:.2f}s/step "
                f"{d['perf/examples_s']:.1f}ex/s {d['perf/tokens_s']/1000:.1f}k tok/s wait {d['perf/data_wait_frac']*100:.0f}% | "
                f"gpu {d.get('gpu/util', 0):.0f}% {d.get('gpu/mem_used_gb', 0):.1f}GB peak {d['cuda/peak_gb']:.1f}GB "
                f"oom {oom_total} | {d['time/hours']:.2f}h ETA {eta/3600:.1f}h")
            L.metrics(step, d, "train")
            acc.clear()
            t_int, wait, n_steps, n_ex, n_tok, n_mb = time.time(), 0.0, 0, 0, 0, 0
            torch.cuda.reset_peak_memory_stats()

        if step % args.eval_every == 0:
            flat = evaluate(step)
            score = flat["select/score"]
            if score > best:
                best, bad = score, 0
                save(os.path.join(out_dir, "best.pt"), step, with_opt=False)
                save_bf16(os.path.join(out_dir, "best_bf16.pt"), step)
                L.p(f"  *** new best select/score {score*100:.2f} -> best.pt")
            else:
                bad += 1
                L.p(f"  no improvement ({bad}); best {best*100:.2f}")
            save(latest, step, with_opt=True)
            save_bf16(os.path.join(out_dir, "latest_model_bf16.pt"), step)
            L.p(f"  mirror to Drive: {'ok' if sync(with_models=True) else 'FAILED (local copies kept)'}")
        elif step % args.save_every == 0:
            save(latest, step, with_opt=True)
            sync(with_models=False)
            L.p(f"  saved latest.pt @ {step}")

        if args.smoke and step == start + args.eval_every and not args.resume:
            # resume round-trip: reload what was just saved and check it matches
            ck = torch.load(latest, map_location="cpu", weights_only=False)
            probe = next(iter(ck["model_state"]))
            same = torch.equal(ck["model_state"][probe], model.trainable_state_dict()[probe])
            L.p(f"  smoke: checkpoint round-trip {'OK' if same else 'MISMATCH'} ({len(ck['model_state'])} tensors)")
            del ck

    L.p(f"stopping: {stop_reason} at step {step}")
    if step % args.eval_every != 0:
        flat = evaluate(step)
        if flat["select/score"] > best:
            best = flat["select/score"]
            save(os.path.join(out_dir, "best.pt"), step, with_opt=False)
            save_bf16(os.path.join(out_dir, "best_bf16.pt"), step)
            L.p(f"  *** new best select/score {best*100:.2f} -> best.pt")
        save(latest, step, with_opt=True)
        save_bf16(os.path.join(out_dir, "latest_model_bf16.pt"), step)
    L.p(f"done. best select/score {best*100:.2f} | total {(time.time()-t_begin)/3600:.2f}h")
    with open(os.path.join(out_dir, "DONE"), "w") as fh:
        fh.write(f"{step} {best}\n")
    if use_wandb:
        import wandb
        wandb.finish()
    synced = sync(with_models=True)
    L.p(f"final mirror to Drive: {'ok' if synced else 'FAILED'}")
    if args.release_vm and not synced:
        L.p("NOT releasing the VM: results are only on its local disk -- copy them off first")
    elif args.release_vm:
        try:
            from google.colab import runtime
            L.p("releasing Colab VM")
            runtime.unassign()
        except Exception as e:
            L.p(f"could not release VM from inside: {e}")


if __name__ == "__main__":
    main()
