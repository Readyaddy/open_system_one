#!/bin/bash
# exp9 step 2 of 2 -- GPU session: preflight checks on the REAL 4B model, then
# the budgeted training run. Every check that can fail cheaply runs before the
# expensive part, and a failed preflight stops the VM immediately.
#
#   wsl -d kali-linux
#   export WANDB_API_KEY=...            # never hard-code it in this file
#   GPU=A100 HOURS=8.5 bash /mnt/d/projects/JEPA/scripts/colab_exp9_train_gpu.sh
#
# Env overrides: GPU (A100|H100|G4|L4), HOURS (wall-clock budget incl. final
# eval), RUN (run name), EXTRA (extra s9_train.py args), SKIP_SMOKE=1.
#
# Preflight (≈15-20 min of GPU time):
#   a. test_branch_mask.py  on the 4B model -- packed branches == independent runs
#   b. test_flex.py         -- FlexAttention exact + faster? then use it, else SDPA
#   c. s9_train.py --smoke  -- 4B, real data: train steps, eval, save, reload
# Training: s9_train.py --max_hours $HOURS --release_vm, checkpoints on Drive,
# resumable (re-running this script resumes from latest.pt).

set -e
source ~/colab-cli-env/bin/activate
GPU=${GPU:-A100}
HOURS=${HOURS:-8.5}
RUN=${RUN:-exp9a}
SESSION=${SESSION:-openjev-exp9-train}
LOCAL_ROOT=/mnt/d/projects/JEPA
C="colab --auth=adc"
if [ -z "$WANDB_API_KEY" ]; then read -r -s -p "WANDB_API_KEY: " WANDB_API_KEY; echo; fi

echo "=== 1. $GPU session ($SESSION) ==="
$C new -s "$SESSION" --gpu "$GPU"
$C drivemount -s "$SESSION"

echo "=== 2. Upload code ==="
TAR=/tmp/openjev_code.tgz
# GNU tar: --exclude must come BEFORE the paths it applies to
tar -czf "$TAR" --exclude='__pycache__' --exclude='*.pt' --exclude='wandb' -C "$LOCAL_ROOT" \
    src scripts experiments/exp7_hybrid_decision/model.py \
    experiments/exp8_s1_rlcd experiments/exp9_scale experiments/jevbench_protocol.py
$C upload -s "$SESSION" "$TAR" /content/openjev_code.tgz

echo "=== 3. Setup: code, data + weights on LOCAL disk, deps ==="
# Drive is only read for the 410 MB data copy (with retries + file-count check) and
# written to by the trainer's retried mirror. Weights come straight from Hugging Face:
# copying 7.5 GB off Drive dropped the FUSE mount mid-copy (Errno 107) on 2026-09-27.
SETUP=$(cat << PYEOF | $C exec -s "$SESSION" --timeout 2400
import os, subprocess, shutil, sys, time
root, drive = '/content/JEPA', '/content/drive/MyDrive/openjev'
os.makedirs(root, exist_ok=True)
subprocess.run(['tar', '-xzf', '/content/openjev_code.tgz', '-C', root], check=True)

src, dst = f'{drive}/data', f'{root}/data'
assert os.path.exists(f'{src}/_stages/ALL_DONE'), 'data not on Drive (MyDrive/openjev/data/_stages/ALL_DONE missing)'
def nfiles(d):
    return sum(len(fs) for _, _, fs in os.walk(d))
for attempt in range(4):
    try:
        if os.path.islink(dst):
            os.unlink(dst)
        shutil.copytree(src, dst, dirs_exist_ok=True)
        if nfiles(dst) >= nfiles(src):
            break
        raise IOError(f'file count {nfiles(dst)} < {nfiles(src)}')
    except Exception as e:
        print('data copy attempt', attempt + 1, 'failed:', str(e)[:200])
        time.sleep(20)
else:
    raise SystemExit('DATA COPY FAILED')
print('data local:', nfiles(dst), 'files')

r = subprocess.run(['pip', 'install', '-q', '-U', 'transformers>=5.3', 'peft', 'wandb', 'nvidia-ml-py',
                    'datasets', 'scikit-learn', 'huggingface_hub'], capture_output=True, text=True)
print('pip rc', r.returncode, r.stderr[-300:])
# Colab ships torchao 0.10; current peft raises ImportError for any torchao < 0.16 even
# though LoRA here never uses it (preflight 4c failure, 2026-09-27).
r = subprocess.run(['pip', 'uninstall', '-y', 'torchao'], capture_output=True, text=True)
print('torchao removed rc', r.returncode)

from huggingface_hub import snapshot_download
for repo in ['Qwen/Qwen3-Embedding-4B', 'Qwen/Qwen3-Embedding-0.6B']:
    d = '/content/hf/' + repo.split('/')[1]
    for attempt in range(3):
        try:
            snapshot_download(repo, local_dir=d)
            break
        except Exception as e:
            print('download retry', attempt + 1, str(e)[:200])
            time.sleep(20)
    st = [f for f in os.listdir(d) if f.endswith('.safetensors')]
    assert os.path.exists(f'{d}/config.json') and os.path.exists(f'{d}/tokenizer.json') and st, f'incomplete {d}'
    print(repo, 'ok', len(st), 'weight files', round(sum(os.path.getsize(f'{d}/{f}') for f in st) / 2**30, 2), 'GB')

r = subprocess.run([sys.executable, '-c', 'import peft, transformers; from transformers import AutoTokenizer; '
                    'AutoTokenizer.from_pretrained("/content/hf/Qwen3-Embedding-4B"); '
                    'print("peft", peft.__version__, "transformers", transformers.__version__, "tokenizer ok")'],
                   capture_output=True, text=True)
print(r.stdout.strip(), r.stderr[-300:] if r.returncode else '')
assert r.returncode == 0, 'peft/tokenizer check failed'
subprocess.run(['wandb', 'login', '--relogin', '$WANDB_API_KEY'], check=True, capture_output=True)
import torch
print('torch', torch.__version__, torch.cuda.get_device_name(0),
      f'{torch.cuda.get_device_properties(0).total_memory/2**30:.0f} GB')
print('SETUP_OK')
PYEOF
)
echo "$SETUP"
if ! echo "$SETUP" | grep -q "^SETUP_OK"; then
  echo "!!! SETUP FAILED -- stopping the VM so no credits burn. !!!"
  $C stop -s "$SESSION"
  exit 1
fi

run_step () {  # $1 = label, $2 = shell command on the VM, $3 = timeout s
  echo "=== $1 ==="
  cat << PYEOF | $C exec -s "$SESSION" --timeout "$3"
import subprocess, os
env = {**os.environ, 'WANDB_API_KEY': '$WANDB_API_KEY', 'PYTHONUNBUFFERED': '1'}
r = subprocess.run("""$2""", shell=True, cwd='/content/JEPA/experiments/exp9_scale', env=env,
                   capture_output=True, text=True)
full = r.stdout + r.stderr
lines = [l for l in full.splitlines() if 'Warning' not in l and 'warn' not in l and '%|' not in l]
# Every training-step line (speed / memory) always shown; the rest is the tail.
steps = [l for l in lines if l.strip().startswith('step ')]
print('\n'.join(steps))
print('\n'.join(lines[-80:]))
# Pass/fail markers computed on the FULL output -- the old check grepped a 5000-char
# tail and missed "round-trip OK" on a smoke run that had actually passed.
print('MARKERS', 'EXACT=%d' % ('EXACT' in full), 'ROUNDTRIP=%d' % ('round-trip OK' in full),
      'DONE=%d' % ('\ndone.' in full or full.startswith('done.')), 'TRACEBACK=%d' % ('Traceback' in full),
      'MIRROR=%d' % ('final mirror to Drive: ok' in full))
print('RC', r.returncode)
PYEOF
}

FAIL=0
if [ -z "$SKIP_SMOKE" ]; then
  OUT=$(run_step "4a. branch-mask exactness on 4B" "python test_branch_mask.py /content/hf/Qwen3-Embedding-4B" 900); echo "$OUT"
  echo "$OUT" | grep -q "MARKERS.*EXACT=1" || FAIL=1
  OUT=$(run_step "4b. FlexAttention check on 4B" "python test_flex.py /content/hf/Qwen3-Embedding-4B" 1200); echo "$OUT"
  ATTN=sdpa
  if echo "$OUT" | grep -q "flex_attention  *fwd+bwd"; then
    S=$(echo "$OUT" | grep "^sdpa " | grep -o "[0-9.]*k tok/s" | grep -o "[0-9.]*")
    F=$(echo "$OUT" | grep "^flex_attention" | grep -o "[0-9.]*k tok/s" | grep -o "[0-9.]*")
    R=$(echo "$OUT" | grep "relative" | grep -o "relative [0-9.e+-]*" | awk '{print $2}')
    if python3 -c "import sys; sys.exit(0 if float('$F')>1.05*float('$S') and float('$R')<2e-2 else 1)" 2>/dev/null; then ATTN=flex_attention; fi
  fi
  echo ">>> attention implementation for training: $ATTN"
  OUT=$(run_step "4c. 4B smoke train (real data)" "python s9_train.py --backbone /content/hf/Qwen3-Embedding-4B --run_name ${RUN}_smoke --out_dir /content/runs/${RUN}_smoke --sync_dir /content/drive/MyDrive/openjev/runs/${RUN}_smoke --smoke --max_steps 30 --eval_every 15 --save_every 15 --log_every 5 --attn $ATTN --no_wandb $EXTRA" 3000); echo "$OUT"
  echo "$OUT" | grep -q "MARKERS.*ROUNDTRIP=1" || FAIL=1
  echo "$OUT" | grep -q "MARKERS.*DONE=1" || FAIL=1
  echo "$OUT" | grep -q "MARKERS.*TRACEBACK=1" && FAIL=1
  echo "$OUT" | grep -q "^RC 0" || FAIL=1
else
  ATTN=${ATTN:-sdpa}
fi

if [ "$FAIL" = "1" ]; then
  echo "!!! PREFLIGHT FAILED -- stopping the VM so no credits burn. Fix locally, then re-run. !!!"
  $C stop -s "$SESSION"
  exit 1
fi

echo "=== 5. Launch training: $RUN on $GPU, budget ${HOURS}h, attn=$ATTN (detached) ==="
cat << PYEOF | $C exec -s "$SESSION" --timeout 60
import subprocess, os
out = '/content/runs/$RUN'                          # local disk: training never depends on Drive
mirror = '/content/drive/MyDrive/openjev/runs/$RUN'   # retried copy target (results + bf16 models)
os.makedirs(out, exist_ok=True)
env = {**os.environ, 'WANDB_API_KEY': '$WANDB_API_KEY', 'PYTHONUNBUFFERED': '1'}
args = ['python3', 's9_train.py', '--backbone', '/content/hf/Qwen3-Embedding-4B', '--run_name', '$RUN',
        '--out_dir', out, '--sync_dir', mirror, '--resume', '--max_hours', '$HOURS', '--attn', '$ATTN',
        '--release_vm',
        '--max_tokens', '16000', '--num_workers', '6'] + '$EXTRA'.split()   # 16k tokens = what the A100 smoke validated
# --resume is always passed: the trainer resumes from local latest.pt, else from the
# Drive mirror (weights only), else starts fresh.
log = open(f'{out}/stdout.log', 'a')
p = subprocess.Popen(args, cwd='/content/JEPA/experiments/exp9_scale', stdout=log, stderr=subprocess.STDOUT,
                     start_new_session=True, env=env)
open('/content/train.pid', 'w').write(str(p.pid))
print('launched pid', p.pid)
PYEOF
echo "=== Launched. Watch: bash $LOCAL_ROOT/scripts/colab_exp9_watch.sh   (W&B project: open-system-one, run $RUN) ==="
echo "=== The run stops itself at the ${HOURS}h budget, evaluates, saves to Drive and releases the VM. ==="
