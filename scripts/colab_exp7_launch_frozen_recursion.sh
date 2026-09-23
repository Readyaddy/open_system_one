#!/bin/bash
# Launches the frozen-backbone recursion/head-only continuation phase --
# see model.py's HybridDecisionModel.freeze_backbone() and train.py's
# --freeze_backbone flag. Loads exp7a's current best-zero-shot weights,
# freezes the backbone entirely, and trains only the entry layer,
# recurrent block, scorer, context codes, and injection/MaxSim components
# at --k_max 6 (depth-conditioned, exp7b-shaped) -- cheap, since there's no
# backbone backward pass or optimizer state for its ~395M params at all.
#
# Runs on its OWN NEW session (jepa-exp7-frozen), not jepa-exp7-train --
# that GPU is fully occupied by exp7a's still-running fine-tuning (26GB+
# used), so this needs separate compute, not to share it. Mounts the SAME
# Google Drive as jepa-exp7-train, so exp7a's checkpoints are visible here
# immediately with no download/transfer step -- --resume points straight
# at exp7a's live exp7_best_zeroshot.pt (kept continuously up to date by
# that other session; the tiny race-condition risk of reading it mid-write
# once every ~25min epoch isn't worth defending against for a one-shot
# manual launch -- colab_exp7_snapshot_best_checkpoint.sh exists if you
# ever want that extra safety instead).
#
# Never touches jepa-exp7-train -- no upload, no exec, nothing -- this
# script only ever talks to the new jepa-exp7-frozen session.
#
# Usage:
#   wsl -d kali-linux
#   bash /mnt/d/projects/JEPA/scripts/colab_exp7_launch_frozen_recursion.sh

set -e
source ~/colab-cli-env/bin/activate

SESSION=jepa-exp7-frozen
LOCAL_ROOT=/mnt/d/projects/JEPA
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export WANDB_API_KEY="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"

check_and_reconnect() {
  if [ "$2" -eq 0 ] && ! echo "$1" | grep -qi "session .* not found"; then
    return 0
  fi
  echo "!!! '$SESSION' unreachable -- auto-reconnecting... !!!"
  if python3 "$SCRIPT_DIR/colab_reconnect.py" "$SESSION"; then
    echo "!!! Reconnected. Resuming. !!!"
    return 0
  else
    echo "!!! Auto-reconnect failed -- run manually: bash $SCRIPT_DIR/colab_reconnect.sh $SESSION --endpoint <endpoint>"
    return 1
  fi
}

echo "=== 1. Creating Colab session ($SESSION) ==="
echo "    Frozen backbone => no backward pass / optimizer state for the ~395M backbone params,"
echo "    so this likely doesn't need a full A100 -- using A100 anyway since it's the confirmed-"
echo "    available tier on this account (T4/L4 would probably also work; not verified)."
colab --auth=adc new -s "$SESSION" --gpu A100

echo "=== 2. Mounting Google Drive (SAME Drive as jepa-exp7-train -- approve the browser prompt) ==="
colab --auth=adc drivemount -s "$SESSION"

echo "=== 3. Creating remote directories ==="
cat << 'PYEOF' | colab --auth=adc exec -s "$SESSION" --timeout 60
import os
for d in ['/content/experiments/exp7_hybrid_decision', '/content/src', '/content/logs',
          '/content/data/intent_corpus', '/content/data/qqp_paraphrase_pairs', '/content/data/mcq_corpus',
          '/content/data/exp7_bool_corpus', '/content/data/exp7_score_corpus',
          '/content/data/exp7_diversity_corpus', '/content/data/exp7_typed_decisions']:
    os.makedirs(d, exist_ok=True)
print('dirs ready')
PYEOF

echo "=== 4. Uploading code (includes the s0 fix + --freeze_backbone, neither ever ran on jepa-exp7-train) ==="
for f in model.py data.py train.py eval.py calibrate.py; do
  colab --auth=adc upload -s "$SESSION" "$LOCAL_ROOT/experiments/exp7_hybrid_decision/$f" "/content/experiments/exp7_hybrid_decision/$f"
done
colab --auth=adc upload -s "$SESSION" "$LOCAL_ROOT/src/local_data.py" /content/src/local_data.py
colab --auth=adc upload -s "$SESSION" "$LOCAL_ROOT/src/dataset_v7.py" /content/src/dataset_v7.py

echo "=== 5. Uploading intent_corpus data ==="
for f in labels.json test.jsonl test_oos.jsonl test_zero_shot.jsonl train.jsonl val.jsonl; do
  colab --auth=adc upload -s "$SESSION" "$LOCAL_ROOT/data/intent_corpus/$f" "/content/data/intent_corpus/$f"
done

echo "=== 6. Reusing cached HF backbone + installing packages ==="
cat << 'PYEOF' | colab --auth=adc exec -s "$SESSION" --timeout 120
import shutil, subprocess, os
cache_backup = '/content/drive/MyDrive/jepa_checkpoints/hf_cache'
if os.path.exists(cache_backup):
    shutil.copytree(cache_backup, os.path.expanduser('~/.cache/huggingface'), dirs_exist_ok=True)
    print('reused cached HF backbone from Drive')
r = subprocess.run(['pip', 'install', '-q', '-U', 'transformers', 'bitsandbytes', 'wandb'],
                    capture_output=True, text=True)
print('deps installed, returncode', r.returncode)
PYEOF

echo "=== 7. Restoring mcq/qqp/exp7-corpora from Drive backup (already built by jepa-exp7-train) ==="
cat << 'PYEOF' | colab --auth=adc exec -s "$SESSION" --timeout 300
import shutil, os
backup_root = '/content/drive/MyDrive/jepa_checkpoints/data_backup'
for name in ['mcq_corpus', 'qqp_paraphrase_pairs', 'exp7_bool_corpus', 'exp7_score_corpus',
             'exp7_diversity_corpus', 'exp7_typed_decisions']:
    src = os.path.join(backup_root, name)
    dst = f'/content/data/{name}'
    if os.path.exists(src):
        shutil.copytree(src, dst, dirs_exist_ok=True)
        print(f'{name}: restored from Drive backup')
    else:
        print(f'{name}: NOT in Drive backup -- that task will be inactive this run')
PYEOF

echo "=== 8. Verifying exp7a has a best-zero-shot checkpoint to resume from ==="
cat << 'PYEOF' | colab --auth=adc exec -s "$SESSION" --timeout 30
import os
p = '/content/drive/MyDrive/jepa_checkpoints/exp7/exp7_best_zeroshot.pt'
if not os.path.exists(p):
    print(f'ERROR: {p} does not exist yet -- exp7a has not saved a best-zeroshot checkpoint.')
    raise SystemExit(1)
print(f'Found: {p}  ({os.path.getsize(p) / 1e6:.1f} MB)')
PYEOF

echo "=== 9. Logging into W&B on the VM ==="
cat << PYEOF | colab --auth=adc exec -s "$SESSION" --timeout 45
import subprocess
subprocess.run(['wandb', 'login', '$WANDB_API_KEY'], check=True)
print('wandb logged in')
PYEOF

echo "=== 10. Launching training (--freeze_backbone --k_max 6, resuming exp7a's live best-zeroshot weights) ==="
cat << PYEOF | colab --auth=adc exec -s "$SESSION" --timeout 60
import subprocess, os
ckpt_dir = '/content/drive/MyDrive/jepa_checkpoints/exp7_frozen'
os.makedirs(ckpt_dir, exist_ok=True)
log = open(os.path.join(ckpt_dir, 'train_full.log'), 'w')
env = os.environ.copy()
env['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
env['WANDB_API_KEY'] = '$WANDB_API_KEY'
source_ckpt = '/content/drive/MyDrive/jepa_checkpoints/exp7/exp7_best_zeroshot.pt'
own_latest = os.path.join(ckpt_dir, 'exp7_frozen_latest.pt')
# --freeze_backbone is passed EVERY launch, resume or not -- requires_grad
# isn't part of a saved state_dict, so it has to be re-applied on every
# process start or a relaunch-after-disconnect would silently un-freeze the
# backbone. --reset_training_state is the one that's launch-specific: only
# the FIRST launch (branching off exp7a's checkpoint, a different set of
# param groups) should discard the checkpoint's optimizer/epoch state: a
# later relaunch of THIS SAME phase should resume its own progress properly.
if os.path.exists(own_latest):
    resume_path, extra_flags = own_latest, ['--freeze_backbone']
    print('relaunching this phase, resuming its own progress from', own_latest)
else:
    resume_path, extra_flags = source_ckpt, ['--freeze_backbone', '--reset_training_state']
    print('first launch: branching off exp7a weights from', source_ckpt)
args = ['python3', '/content/experiments/exp7_hybrid_decision/train.py',
     '--k_max', '6', '--epochs', '20', '--steps_per_epoch', '500',
     '--batch_size', '32', '--grad_accum', '2', '--option_chunk_size', '256', '--patience', '6',
     '--resume', resume_path] + extra_flags + [
     '--ckpt_prefix', 'exp7_frozen',
     '--wandb_project', 'jepa-exp7', '--wandb_run_id', 'exp7_frozen',
     '--ckpt_dir', ckpt_dir]
proc = subprocess.Popen(args, cwd='/content/experiments/exp7_hybrid_decision',
    stdout=log, stderr=subprocess.STDOUT, start_new_session=True, env=env)
with open('/content/logs/train_full.pid', 'w') as f:
    f.write(str(proc.pid))
print('launched, pid', proc.pid, 'args', args)
PYEOF

echo ""
echo "=== Launched. Switching to live monitor (Ctrl+C to stop watching -- training keeps running). ==="
echo "=== Also viewable at https://wandb.ai (project: jepa-exp7, run: exp7_frozen) ==="
echo "=== batch_size/option_chunk_size below are exp7a's KNOWN-WORKING settings as a starting"
echo "=== point -- likely conservative here since the backbone carries no backward/optimizer"
echo "=== cost anymore; watch gpu_mem and push both up once you've seen a few real steps. ==="
sleep 5

set +e
while true; do
  clear
  echo "=== $(date) ==="
  OUTPUT=$(cat << 'PYEOF' | colab --auth=adc exec -s "$SESSION" --timeout 30 2>&1
import subprocess
print(subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu',
                       '--format=csv'], capture_output=True, text=True).stdout)
pid = open('/content/logs/train_full.pid').read().strip()
ps = subprocess.run(['ps', '-p', pid, '-o', 'pid,stat,etimes'], capture_output=True, text=True).stdout
print(ps if len(ps.strip().splitlines()) > 1 else f'PROCESS {pid} NOT FOUND -- may have crashed/disconnected')
lines = open('/content/drive/MyDrive/jepa_checkpoints/exp7_frozen/train_full.log').read().splitlines()
step_lines = [l for l in lines if l.strip().startswith('epoch') and 'step' in l][-10:]
summary_lines = [l for l in lines if l.strip().startswith('epoch') and 'step' not in l][-3:]
other_recent = [l for l in lines if not l.strip().startswith('epoch')][-8:]
print('\n'.join(other_recent))
print('--- recent per-step progress ---')
print('\n'.join(step_lines))
print('--- recent epoch summaries ---')
print('\n'.join(summary_lines))
PYEOF
)
  STATUS=$?
  echo "$OUTPUT"
  check_and_reconnect "$OUTPUT" "$STATUS"
  sleep 30
done
