"""Evaluation script for JevBench Hard-111 benchmark dataset.
Runs Open System-1 model on hard_111.jsonl (111 hard-tier decision routing tasks).
"""
import json
import os
import sys
import torch
import torch.nn.functional as F

# Torchvision stub wrapper
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
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import jevbench_protocol as JB  # noqa: E402

from model import HybridDecisionModel, PackedSequenceBuilder, PackedExample, get_tokenizer, BACKBONE

def run_hard111_eval(ckpt_path: str, data_path: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    tok = get_tokenizer(BACKBONE)
    builder = PackedSequenceBuilder(tok, budget_total=8192, l_context=7168)
    
    ckpt = torch.load(ckpt_path, map_location=device)
    k_max = ckpt.get("k_max", 6)
    use_maxsim = ckpt.get("use_maxsim", False)
    
    model = HybridDecisionModel(backbone=BACKBONE, mask_token_id=tok.mask_token_id,
                                k_max=k_max, use_maxsim=use_maxsim)
    state_dict = ckpt.get("model", ckpt.get("model_state_dict", ckpt))
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    
    print(f"Loaded checkpoint: {ckpt_path}")
    
    with open(data_path, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
        
    print(f"Loaded {len(rows)} items from {data_path}")
    
    correct = 0
    total = 0
    
    for idx, row in enumerate(rows):
        # Official JevBench protocol (criteria as options, noul -> bool, JSON
        # states) -- shared with every other evaluator via jevbench_protocol.
        x = JB.render(row, "hard")
        ex = PackedExample(
            context=x["context"],
            instructions=x["instructions"],
            option_texts=x["option_texts"],
            qtype=x["qtype"],
            answer_idx=x["answer_idx"],
            use_text=True,
            use_vector=True
        )
        answer_idx = x["answer_idx"]

        batch = builder.build_batch([ex], device)
        
        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits_per_depth = model(tok, batch, device)
                logits = logits_per_depth[-1] # final depth
                pred = logits.argmax(dim=-1).item()
                
        is_correct = (pred == answer_idx)
        if is_correct:
            correct += 1
        total += 1
        
        if (idx + 1) % 10 == 0 or (idx + 1) == len(rows):
            print(f"  Processed {idx + 1}/{len(rows)} | Current Acc: {correct / total:.4f} ({correct}/{total})", flush=True)
            
    final_acc = correct / total if total > 0 else 0.0
    print(f"\n==========================================")
    print(f"JevBench Hard-111 Evaluation Accuracy: {final_acc:.4f} ({correct}/{total})")
    print(f"==========================================")
    return final_acc

if __name__ == "__main__":
    ckpt = sys.argv[1] if len(sys.argv) > 1 else r"D:\projects\JEPA\checkpoints\exp7c_local_nomaxsim_v2\exp7c_nomaxsim_v2_best_zeroshot.pt"
    data = sys.argv[2] if len(sys.argv) > 2 else r"D:\projects\JEPA\data\hard_111.jsonl"
    run_hard111_eval(ckpt, data)
