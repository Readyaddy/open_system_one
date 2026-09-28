"""Can exp9 use FlexAttention for the branch mask? Checks (1) Qwen3 in this
transformers accepts a BlockMask, (2) outputs match the SDPA 4D-mask path,
(3) speed on a realistic micro-batch."""
import sys, time, torch
sys.path.insert(0, "../exp8_s1_rlcd")
import s1_data as SD
from s9_model import Packer9, get_tokenizer
from transformers import AutoModel
from torch.nn.attention.flex_attention import create_block_mask
from torch.amp import autocast

name = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-Embedding-0.6B"
dev = torch.device("cuda")
tok = get_tokenizer(name)
C = SD.load_corpora()
smp = SD.TrainSampler(C, SD.DataCfg(p_bool_desc=0.5), seed=1)
b, _ = Packer9(tok, max_tokens=8000).pack([smp.sample() for _ in range(16)])[0]
B, L = b.input_ids.shape
print("micro-batch", (B, L), "options", int(b.valid_mask.sum()), "density of allowed attention",
      f"{b.attn4d.float().mean().item():.3f}")
b = b.to(dev)
seg = torch.full((B, L), -1, device=dev)
# rebuild segment ids from the 4D mask's structure is awkward -> derive from position/prefix info:
# prefix tokens -> 0; branch tokens -> branch index via mark positions.
seg[b.prefix_mask] = 0
for i in range(B):
    marks = b.mark_pos[i][b.valid_mask[i]].tolist()
    reads = b.read_pos[i][b.valid_mask[i]].tolist()
    for j, (m0, r0) in enumerate(zip(marks, reads), 1):
        seg[i, m0:r0 + 1] = j

def mask_mod(bi, h, q, k):
    sq, sk = seg[bi, q], seg[bi, k]
    return (k <= q) & (((sk == 0) | (sk == sq)) & (sk >= 0) | (q == k))

bm = create_block_mask(mask_mod, B, None, L, L, device=dev)
print("flex block mask sparsity:", f"{bm.sparsity():.1f}%")

res = {}
for impl, mask in (("sdpa", b.attn4d), ("flex_attention", bm)):
    try:
        mdl = AutoModel.from_pretrained(name, dtype=torch.bfloat16, attn_implementation=impl).to(dev)
        mdl.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        mdl.train()
        def step():
            emb = mdl.get_input_embeddings()(b.input_ids).detach().requires_grad_(True)
            H = mdl(inputs_embeds=emb, attention_mask=mask, position_ids=b.position_ids).last_hidden_state
            H.float().pow(2).mean().backward()
            return H
        H = step(); torch.cuda.synchronize()
        t = time.time()
        for _ in range(3):
            step()
        torch.cuda.synchronize()
        dt = (time.time() - t) / 3
        with torch.no_grad():
            mdl.eval()
            Hr = mdl(inputs_embeds=mdl.get_input_embeddings()(b.input_ids), attention_mask=mask,
                     position_ids=b.position_ids).last_hidden_state
        res[impl] = Hr.float()
        print(f"{impl:15s} fwd+bwd {dt*1000:6.0f} ms  -> {B*L/dt/1000:5.1f}k tok/s (padded)")
        del mdl
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"{impl:15s} FAILED: {type(e).__name__}: {str(e)[:300]}")
if len(res) == 2:
    rd = b.read_pos.clamp(min=0)
    a = res["sdpa"].gather(1, rd.unsqueeze(-1).expand(-1, -1, res["sdpa"].size(-1)))
    f = res["flex_attention"].gather(1, rd.unsqueeze(-1).expand(-1, -1, a.size(-1)))
    v = b.valid_mask.unsqueeze(-1)
    print(f"readout agreement flex vs sdpa: max abs diff {((a - f).abs() * v).max().item():.4f} "
          f"(bf16; relative {((a - f).abs() * v).max().item() / a.abs().max().item():.2e})")
