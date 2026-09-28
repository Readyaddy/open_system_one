"""Where does exp9's step time go? Times each component on a realistic batch."""
import sys, time, torch
sys.path.insert(0, "../exp8_s1_rlcd")
import s1_data as SD
from s9_model import S9DecisionModel, Packer9, get_tokenizer
from torch.amp import autocast

name = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-Embedding-0.6B"
head = sys.argv[2] if len(sys.argv) > 2 else "rethink"
dev = torch.device("cuda")
tok = get_tokenizer(name)
C = SD.load_corpora()
cfg = SD.DataCfg(p_bool_desc=0.5)
smp = SD.TrainSampler(C, cfg, seed=1)
packer = Packer9(tok, max_tokens=8000)
mbs = packer.pack([smp.sample() for _ in range(16)])
print(f"{len(mbs)} micro-batches, shapes {[tuple(b.input_ids.shape) for b, _ in mbs]}, options {[int(b.valid_mask.sum()) for b, _ in mbs]}")
m = S9DecisionModel(name, head=head, block_layers=4 if head == "rethink" else 2).to(dev)
m.train()
pc = m.param_counts(); print(f"params total {pc['total']/1e9:.3f}B trainable {pc['trainable']/1e6:.0f}M")

def timeit(fn, n=3):
    fn(); torch.cuda.synchronize()
    t = time.time()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t) / n

b = mbs[-1][0].to(dev)
print("batch", tuple(b.input_ids.shape))
t_emb = timeit(lambda: m.embed_options(b.opt_ids, b.opt_mask))
def bb(mask4):
    emb = m.input_embeddings()(b.input_ids).detach().requires_grad_(True)
    with autocast("cuda", dtype=torch.bfloat16):
        if mask4:
            H = m.backbone(inputs_embeds=emb, attention_mask=b.attn4d, position_ids=b.position_ids).last_hidden_state
        else:
            H = m.backbone(inputs_embeds=emb, attention_mask=torch.ones_like(b.input_ids), position_ids=b.position_ids).last_hidden_state
    H.float().pow(2).mean().backward()
t4 = timeit(lambda: bb(True)); t2 = timeit(lambda: bb(False))
def full():
    with autocast("cuda", dtype=torch.bfloat16):
        outs = m(b, all_depths=True)
    sum(o.float().logsumexp(-1).mean() for o in outs).backward()
tf = timeit(full)
print(f"option embedding pass : {t_emb*1000:7.0f} ms")
print(f"backbone fwd+bwd 4D   : {t4*1000:7.0f} ms")
print(f"backbone fwd+bwd plain: {t2*1000:7.0f} ms   (4D mask overhead x{t4/t2:.2f})")
print(f"full model fwd+bwd    : {tf*1000:7.0f} ms   (head+inject ~{(tf-t4-t_emb)*1000:.0f} ms)")
print(f"peak mem {torch.cuda.max_memory_allocated()/2**30:.1f} GB")
