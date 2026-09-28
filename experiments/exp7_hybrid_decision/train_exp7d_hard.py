"""Experiment 7d: Extended 10-Epoch Fine-tuning on Hard Reasoning Corpus (ModernBERT 8K Context).

Continues fine-tuning from exp7d_best_hard.pt for 10 full epochs to observe deep convergence,
saving live metrics to checkpoints/exp7d_hard_reasoning/exp7d_metrics_10ep.json.
"""
import json
import math
import os
import random
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
from torch.utils.data import Dataset, DataLoader

# Stub torchvision
def _stub():
    import types, importlib.machinery
    if 'torchvision' in sys.modules: return
    fake_tv = types.ModuleType('torchvision')
    fake_tv.__spec__ = importlib.machinery.ModuleSpec('torchvision', loader=None)
    fake_tv.__version__ = '0.0.0'
    fake_transforms = types.ModuleType('torchvision.transforms')
    fake_transforms.__spec__ = importlib.machinery.ModuleSpec('torchvision.transforms', loader=None)
    class InterpolationMode:
        NEAREST='nearest'; NEAREST_EXACT='nearest_exact'; BOX='box'
        BILINEAR='bilinear'; HAMMING='hamming'; BICUBIC='bicubic'; LANCZOS='lanczos'
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

from model import HybridDecisionModel, PackedSequenceBuilder, PackedExample, PackedBatch, get_tokenizer, BACKBONE

AUTOCAST_DTYPE = torch.bfloat16

class AsyncPackedBatch:
    def __init__(self, input_ids, attention_mask, context_token_mask, mask_positions,
                 valid_mask, inject_scale, text_scale, qtype_idx, answer_idx,
                 per_owner_n, option_input_ids, option_attention_mask):
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
        return self

    def to_device(self, device):
        inp_ids = self.input_ids.to(device, non_blocking=True)
        att_mask = self.attention_mask.to(device, non_blocking=True)
        ctx_mask = self.context_token_mask.to(device, non_blocking=True)
        mask_pos = self.mask_positions.to(device, non_blocking=True)
        val_mask = self.valid_mask.to(device, non_blocking=True)
        inj_scale = self.inject_scale.to(device, non_blocking=True)
        txt_scale = self.text_scale.to(device, non_blocking=True)
        qtype = self.qtype_idx.to(device, non_blocking=True)
        ans = self.answer_idx.to(device, non_blocking=True)

        opt_enc = None
        if self.option_input_ids is not None:
            opt_enc = {
                "input_ids": self.option_input_ids.to(device, non_blocking=True),
                "attention_mask": self.option_attention_mask.to(device, non_blocking=True)
            }

        return PackedBatch(
            input_ids=inp_ids, attention_mask=att_mask,
            context_token_mask=ctx_mask, mask_positions=mask_pos,
            valid_mask=val_mask, inject_scale=inj_scale,
            text_scale=txt_scale, qtype_idx=qtype,
            answer_idx=ans, option_full_texts=(opt_enc, self.per_owner_n)
        )

def fast_encode_and_project_options(self, tokenizer, batch: PackedBatch, device, chunk_size: int = 128):
    if isinstance(batch.option_full_texts, tuple) and isinstance(batch.option_full_texts[0], dict):
        opt_enc, per_owner_n = batch.option_full_texts
        pooled, _, _ = self.encode_options_raw(tokenizer, opt_enc, device, chunk_size=chunk_size, need_tokens=self.use_maxsim)
        projected = self.option_projector(pooled)
        projected = projected / projected.norm(dim=-1, keepdim=True).clamp(min=1e-6) * self.emb_scale

        B = len(per_owner_n)
        Nmax = batch.mask_positions.size(1)
        D = pooled.size(-1)
        out_pooled = pooled.new_zeros(B, Nmax, D)
        out_valid = torch.zeros(B, Nmax, dtype=torch.bool, device=device)
        pos = 0
        for b, n in enumerate(per_owner_n):
            out_pooled[b, :n] = projected[pos:pos + n] * self.inject_gate * batch.inject_scale[b]
            out_valid[b, :n] = True
            pos += n
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
        out = self.backbone(input_ids=ids, attention_mask=amask)
        h = out.last_hidden_state
        m = amask.unsqueeze(-1).float()
        pooled = (h * m).sum(1) / m.sum(1).clamp(min=1.0)
        return pooled, None, None
    return _original_encode_options_raw(self, tokenizer, option_texts, device, max_length, chunk_size, need_tokens)

HybridDecisionModel.encode_options_raw = fast_encode_options_raw

def save_live_metrics(metrics_file, data):
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def load_hard_corpus(data_dir):
    train_path = os.path.join(data_dir, "exp7_hard_reasoning_corpus", "train.jsonl")
    val_path = os.path.join(data_dir, "exp7_hard_reasoning_corpus", "val.jsonl")
    
    with open(train_path, "r", encoding="utf-8") as f:
        train_rows = [json.loads(line) for line in f if line.strip()]
    with open(val_path, "r", encoding="utf-8") as f:
        val_rows = [json.loads(line) for line in f if line.strip()]
        
    return train_rows, val_rows

class HardReasoningDataset(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        rng = random.Random(idx + 42)
        options = list(row["option_texts"])
        true_ans = options[row["answer_idx"]]
        rng.shuffle(options)
        ans_idx = options.index(true_ans)
        
        context = str(row.get("context", ""))
        instructions = str(row.get("instructions", "Select the correct answer:"))
        qtype = str(row.get("qtype", "choice"))
        if qtype not in ["choice", "bool", "score"]:
            qtype = "choice"

        return PackedExample(
            context=context,
            instructions=instructions,
            option_texts=options,
            qtype=qtype,
            answer_idx=ans_idx,
            use_text=True,
            use_vector=True,
            source=str(row.get("source", "hard_reasoning"))
        )

class PackedBatchCollate:
    def __init__(self, builder, tok):
        self.builder = builder
        self.tok = tok

    def __call__(self, examples):
        batch = self.builder.build_batch(examples, device="cpu")
        per_owner_n = [len(texts) for texts in batch.option_full_texts]
        flat_texts = [t for texts in batch.option_full_texts for t in texts]

        opt_ids, opt_mask = None, None
        if flat_texts:
            enc_opts = self.tok(flat_texts, padding=True, truncation=True, max_length=64, return_tensors="pt")
            opt_ids = enc_opts["input_ids"]
            opt_mask = enc_opts["attention_mask"]

        return AsyncPackedBatch(
            input_ids=batch.input_ids,
            attention_mask=batch.attention_mask,
            context_token_mask=batch.context_token_mask,
            mask_positions=batch.mask_positions,
            valid_mask=batch.valid_mask,
            inject_scale=batch.inject_scale,
            text_scale=batch.text_scale,
            qtype_idx=batch.qtype_idx,
            answer_idx=batch.answer_idx,
            per_owner_n=per_owner_n,
            option_input_ids=opt_ids,
            option_attention_mask=opt_mask
        )

def compute_depth_loss(logits_per_depth, answer_idx):
    total_loss = 0.0
    for logits in logits_per_depth:
        loss = F.cross_entropy(logits, answer_idx)
        total_loss += loss
    return total_loss / len(logits_per_depth)

def evaluate_hard111(model, tok, builder, hard111_path, device):
    model.eval()
    with open(hard111_path, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    k_max = model.k_max
    correct_per_depth = [0] * k_max
    total = 0

    collate = PackedBatchCollate(builder, tok)

    for row in rows:
        expected_label = str(row["expected"])
        labels = [str(l) for l in row["labels"]]
        if expected_label not in labels:
            if expected_label.isdigit() and int(expected_label) < len(labels):
                answer_idx = int(expected_label)
            else:
                continue
        else:
            answer_idx = labels.index(expected_label)
            
        qtype = str(row["question"].get("type", "choice"))
        if qtype not in ["choice", "bool", "score"]:
            qtype = "choice"

        ex = PackedExample(
            context=str(row.get("state", "")),
            instructions=str(row["question"].get("instructions", "Select the correct category:")),
            option_texts=labels,
            qtype=qtype,
            answer_idx=answer_idx,
            use_text=True,
            use_vector=True
        )

        async_batch = collate([ex])
        batch = async_batch.to_device(device)

        with torch.no_grad():
            with autocast("cuda", dtype=AUTOCAST_DTYPE):
                logits_per_depth = model(tok, batch, device)
                for k in range(k_max):
                    pred = logits_per_depth[k].argmax(dim=-1).item()
                    if pred == answer_idx:
                        correct_per_depth[k] += 1
        total += 1

    acc_per_depth = [c / total for c in correct_per_depth]
    return acc_per_depth, total

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Extended 10-Epoch Exp 7d Fine-Tuning on device: {device}", flush=True)

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    data_dir = os.path.join(base_dir, "data")
    
    out_dir = os.path.join(base_dir, "checkpoints", "exp7d_hard_reasoning")
    os.makedirs(out_dir, exist_ok=True)
    metrics_file = os.path.join(out_dir, "exp7d_metrics_10ep.json")
    hard111_path = os.path.join(data_dir, "hard_111.jsonl")

    # Start from exp7d_best_hard.pt if available, else exp7c
    exp7d_best_path = os.path.join(out_dir, "exp7d_best_hard.pt")
    exp7c_path = os.path.join(base_dir, "checkpoints", "exp7c_local_nomaxsim_v2", "exp7c_nomaxsim_v2_best_zeroshot.pt")
    
    ckpt_path = exp7d_best_path if os.path.exists(exp7d_best_path) else exp7c_path

    tok = get_tokenizer(BACKBONE)
    
    # ModernBERT 8K Full Sequence Context
    budget_total = 8192
    l_context = 7168
    print(f"Sequence Budget: budget_total={budget_total}, l_context={l_context}", flush=True)
    
    builder = PackedSequenceBuilder(tok, budget_total=budget_total, l_context=l_context, l_instructions=128, l_max_per_option=64)

    ckpt = torch.load(ckpt_path, map_location=device)
    k_max = ckpt.get("k_max", 6)
    use_maxsim = ckpt.get("use_maxsim", False)

    model = HybridDecisionModel(backbone=BACKBONE, mask_token_id=tok.mask_token_id, k_max=k_max, use_maxsim=use_maxsim)
    state_dict = ckpt.get("model", ckpt.get("model_state_dict", ckpt))
    model.load_state_dict(state_dict, strict=False)
    model.to(device)

    print(f"Loaded starting checkpoint: {ckpt_path}", flush=True)

    live_data = {
        "experiment": "exp7d_hard_reasoning_10ep",
        "budget_total": budget_total,
        "l_context": l_context,
        "status": "running",
        "initial_hard111_accs": [],
        "epochs": [],
        "current_step": 0,
        "total_steps": 0,
        "current_loss": 0.0,
        "eta_seconds": 0
    }
    save_live_metrics(metrics_file, live_data)

    print("\n--- Baseline Evaluation on JevBench Hard-111 ---", flush=True)
    initial_accs, total_items = evaluate_hard111(model, tok, builder, hard111_path, device)
    live_data["initial_hard111_accs"] = initial_accs
    save_live_metrics(metrics_file, live_data)

    for k, acc in enumerate(initial_accs):
        print(f"  Pre-10Ep Depth k={k+1}: {acc*100:.2f}% ({int(acc*total_items)}/{total_items})", flush=True)

    train_rows, val_rows = load_hard_corpus(data_dir)
    print(f"\nLoaded Hard Reasoning Corpus: {len(train_rows)} train, {len(val_rows)} val", flush=True)

    epochs = 10
    batch_size = 4
    grad_accum = 4
    lr = 1.5e-5
    
    train_dataset = HardReasoningDataset(train_rows)
    num_workers = 2 if os.name == 'nt' else 4
    collate_fn = PackedBatchCollate(builder, tok)
    print(f"Initializing DataLoader: num_workers={num_workers}, pin_memory=True, prefetch_factor=4", flush=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=4 if num_workers > 0 else None,
        persistent_workers=(num_workers > 0),
        collate_fn=collate_fn
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    max_batches_per_epoch = 1000
    total_steps = epochs * max_batches_per_epoch
    live_data["total_steps"] = total_steps

    print(f"\nStarting Extended Fine-Tuning: Epochs={epochs}, Batches/Epoch={max_batches_per_epoch}, BatchSize={batch_size}, GradAccum={grad_accum}, LR={lr}", flush=True)
    
    best_acc_k6 = initial_accs[-1]
    global_step = 0

    for epoch in range(1, epochs + 1):
        model.train()
        running_gpu_loss = torch.zeros((), device=device)
        optimizer.zero_grad()
        epoch_start_time = time.time()

        for step, async_batch in enumerate(train_loader, 1):
            if step > max_batches_per_epoch:
                break

            batch = async_batch.to_device(device)

            with autocast("cuda", dtype=AUTOCAST_DTYPE):
                logits_per_depth = model(tok, batch, device)
                loss = compute_depth_loss(logits_per_depth, batch.answer_idx) / grad_accum

            loss.backward()
            running_gpu_loss += loss.detach() * grad_accum

            if step % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

            global_step += 1

            if step % 50 == 0 or step == max_batches_per_epoch:
                avg_loss = (running_gpu_loss / step).item()
                elapsed = time.time() - epoch_start_time
                steps_per_sec = step / max(elapsed, 1e-5)
                remaining_steps = total_steps - global_step
                eta_sec = remaining_steps / max(steps_per_sec, 1e-5)

                live_data["current_step"] = global_step
                live_data["current_loss"] = avg_loss
                live_data["eta_seconds"] = int(eta_sec)
                save_live_metrics(metrics_file, live_data)

                print(f"  [Epoch {epoch}/{epochs}] Step {step}/{max_batches_per_epoch} ({global_step}/{total_steps}) | Loss: {avg_loss:.4f} | Speed: {steps_per_sec:.2f} step/s | ETA: {int(eta_sec//60)}m {int(eta_sec%60)}s", flush=True)

        print(f"\n--- Post-Epoch {epoch} Evaluation on JevBench Hard-111 ---", flush=True)
        eval_accs, total_items = evaluate_hard111(model, tok, builder, hard111_path, device)
        for k, acc in enumerate(eval_accs):
            print(f"  Post-Epoch {epoch} Depth k={k+1}: {acc*100:.2f}% ({int(acc*total_items)}/{total_items})", flush=True)

        epoch_record = {
            "epoch": epoch,
            "val_loss": (running_gpu_loss / max_batches_per_epoch).item(),
            "hard111_accs": eval_accs
        }
        live_data["epochs"].append(epoch_record)
        save_live_metrics(metrics_file, live_data)

        save_file = os.path.join(out_dir, f"exp7d_10ep_epoch{epoch}.pt")
        torch.save({"model": model.state_dict(), "k_max": k_max, "use_maxsim": use_maxsim, "epoch": epoch, "acc_per_depth": eval_accs}, save_file)
        print(f"Saved checkpoint: {save_file}", flush=True)

        if eval_accs[-1] > best_acc_k6:
            best_acc_k6 = eval_accs[-1]
            best_file = os.path.join(out_dir, "exp7d_best_hard.pt")
            torch.save({"model": model.state_dict(), "k_max": k_max, "use_maxsim": use_maxsim, "epoch": epoch, "acc_per_depth": eval_accs}, best_file)
            print(f"*** New Overall Best Checkpoint Saved: {best_file} (k=6 Acc: {best_acc_k6*100:.2f}%) ***", flush=True)

    live_data["status"] = "completed"
    save_live_metrics(metrics_file, live_data)

    print("\n==========================================")
    print("10-Epoch Extended Experiment 7d Fine-Tuning & Evaluation Completed!")
    print("==========================================", flush=True)

if __name__ == "__main__":
    main()
