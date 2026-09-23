#!/bin/bash
# Complete, from-scratch setup + launch + LIVE MONITORING, meant to be
# run ONCE from a single terminal you keep open. Run this only after
# terminating any old sessions yourself (website: Runtime > Manage
# sessions), so this creates exactly one clean session.
#
# Usage:
#   wsl -d kali-linux
#   bash /mnt/d/projects/JEPA/scripts/colab_full_setup_and_launch.sh
#
# It pauses once for you to approve the Drive OAuth prompt in your
# browser (drivemount is inherently interactive) -- everything else is
# automatic. After training launches, this terminal switches into a
# live monitor: GPU utilization/memory + the tail of the training log,
# refreshing every 30s, right here -- no need to check wandb or ask
# Claude. Press Ctrl+C to stop watching (training keeps running on the
# VM regardless -- it's a separate detached process).

set -e
source ~/colab-cli-env/bin/activate

SESSION=jepa-train
LOCAL_ROOT=/mnt/d/projects/JEPA
export WANDB_API_KEY="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"

echo "=== 1. Creating Colab session ($SESSION, A100) ==="
colab --auth=adc new -s "$SESSION" --gpu A100

echo "=== 2. Mounting Google Drive (approve the browser prompt when it appears) ==="
colab --auth=adc drivemount -s "$SESSION"

echo "=== 3. Creating remote directories ==="
cat << 'PYEOF' | colab --auth=adc exec -s "$SESSION" --timeout 60
import os
for d in ['/content/experiments/exp6_diverse_data_qqp_aux', '/content/src', '/content/scripts', '/content/logs',
          '/content/data/intent_corpus', '/content/data/qqp_paraphrase_pairs', '/content/data/mcq_corpus']:
    os.makedirs(d, exist_ok=True)
print('dirs ready')
PYEOF

echo "=== 4. Uploading code ==="
colab --auth=adc upload -s "$SESSION" "$LOCAL_ROOT/experiments/exp6_diverse_data_qqp_aux/train.py" /content/experiments/exp6_diverse_data_qqp_aux/train.py
colab --auth=adc upload -s "$SESSION" "$LOCAL_ROOT/experiments/exp6_diverse_data_qqp_aux/model.py" /content/experiments/exp6_diverse_data_qqp_aux/model.py
colab --auth=adc upload -s "$SESSION" "$LOCAL_ROOT/src/local_data.py" /content/src/local_data.py
colab --auth=adc upload -s "$SESSION" "$LOCAL_ROOT/src/dataset_v7.py" /content/src/dataset_v7.py
colab --auth=adc upload -s "$SESSION" "$LOCAL_ROOT/src/mcq_dataset.py" /content/src/mcq_dataset.py

echo "=== 5. Uploading intent_corpus data ==="
for f in labels.json test.jsonl test_oos.jsonl test_zero_shot.jsonl train.jsonl val.jsonl; do
  colab --auth=adc upload -s "$SESSION" "$LOCAL_ROOT/data/intent_corpus/$f" "/content/data/intent_corpus/$f"
done

echo "=== 6. Copying mcq/qqp data from Drive backup + installing packages ==="
cat << 'PYEOF' | colab --auth=adc exec -s "$SESSION" --timeout 90
import shutil, subprocess, os
shutil.copytree('/content/drive/MyDrive/jepa_checkpoints/data_backup/mcq_corpus', '/content/data/mcq_corpus', dirs_exist_ok=True)
shutil.copytree('/content/drive/MyDrive/jepa_checkpoints/data_backup/qqp_paraphrase_pairs', '/content/data/qqp_paraphrase_pairs', dirs_exist_ok=True)
print('mcq:', os.listdir('/content/data/mcq_corpus'))
print('qqp:', os.listdir('/content/data/qqp_paraphrase_pairs'))
r = subprocess.run(['pip', 'install', '-q', 'bitsandbytes', 'wandb'], capture_output=True, text=True)
print('bnb + wandb installed', r.returncode)
PYEOF

echo "=== 7. Logging into W&B on the VM ==="
cat << PYEOF | colab --auth=adc exec -s "$SESSION" --timeout 45
import subprocess
subprocess.run(['wandb', 'login', '$WANDB_API_KEY'], check=True)
print('wandb logged in')
PYEOF

echo "=== 8. Checking for an existing checkpoint to resume from ==="
RESUME_FLAG=""
cat << 'PYEOF' | colab --auth=adc exec -s "$SESSION" --timeout 45
import os
p = '/content/drive/MyDrive/jepa_checkpoints/exp6/exp6_latest.pt'
print('EXISTS' if os.path.exists(p) else 'MISSING')
PYEOF

echo "=== 9. Launching training ==="
cat << PYEOF | colab --auth=adc exec -s "$SESSION" --timeout 60
import subprocess, os
ckpt_dir = '/content/drive/MyDrive/jepa_checkpoints/exp6'
os.makedirs(ckpt_dir, exist_ok=True)
log = open(os.path.join(ckpt_dir, 'train_full.log'), 'w')
env = os.environ.copy()
env['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
env['WANDB_API_KEY'] = '$WANDB_API_KEY'
resume_path = '/content/drive/MyDrive/jepa_checkpoints/exp6/exp6_latest.pt'
args = ['python3', '/content/experiments/exp6_diverse_data_qqp_aux/train.py',
     '--epochs', '30', '--batch_size', '384', '--qqp_batch_size', '384', '--mcq_batch_size', '224',
     '--grad_accum', '1', '--outcome_chunk_size', '255',
     '--qqp_every_n_steps', '1', '--mcq_every_n_steps', '1', '--patience', '8',
     '--intent_options', '50',
     '--wandb_project', 'jepa-exp6', '--wandb_run_id', 'exp6',
     '--ckpt_dir', ckpt_dir]
if os.path.exists(resume_path):
    args += ['--resume', resume_path]
    print('resuming from', resume_path)
else:
    print('no existing checkpoint -- starting fresh')
proc = subprocess.Popen(args, cwd='/content/experiments/exp6_diverse_data_qqp_aux',
    stdout=log, stderr=subprocess.STDOUT, start_new_session=True, env=env)
with open('/content/logs/train_full.pid', 'w') as f:
    f.write(str(proc.pid))
print('launched, pid', proc.pid)
PYEOF

echo ""
echo "=== Training launched. Switching to live monitor (Ctrl+C to stop watching -- training keeps running). ==="
echo "=== Also viewable anytime at https://wandb.ai (project: jepa-exp6) ==="
sleep 5

set +e  # from here on, a transient colab exec failure during monitoring
        # should NOT kill the whole script -- just show the error and retry
while true; do
  clear
  echo "=== $(date) ==="
  cat << 'PYEOF' | colab --auth=adc exec -s "$SESSION" --timeout 30 2>&1
import subprocess
print(subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu',
                       '--format=csv'], capture_output=True, text=True).stdout)
pid = open('/content/logs/train_full.pid').read().strip()
ps = subprocess.run(['ps', '-p', pid, '-o', 'pid,stat,etimes'], capture_output=True, text=True).stdout
print(ps if len(ps.strip().splitlines()) > 1 else f'PROCESS {pid} NOT FOUND -- may have crashed/disconnected')
print('--- last 15 lines of training log (per-step progress + latest epoch summary) ---')
lines = open('/content/drive/MyDrive/jepa_checkpoints/exp6/train_full.log').read().splitlines()
step_lines = [l for l in lines if l.strip().startswith('epoch') and 'step' in l][-10:]
summary_lines = [l for l in lines if l.strip().startswith('epoch') and 'step' not in l][-3:]
other_recent = [l for l in lines if not l.strip().startswith('epoch')][-5:]
print('\n'.join(other_recent))
print('--- recent per-step progress ---')
print('\n'.join(step_lines))
print('--- recent epoch summaries ---')
print('\n'.join(summary_lines))
PYEOF
  sleep 30
done
