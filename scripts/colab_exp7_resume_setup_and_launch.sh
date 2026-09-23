#!/bin/bash
# Resumes exp7 setup from step 3 onward -- use this when `colab new` and
# `colab drivemount` already succeeded (e.g. the previous script hit a
# transient connection blip partway through) so you don't redo those steps
# or risk creating a duplicate session. Mirrors exp6's
# colab_resume_setup_and_launch.sh.

set -e
source ~/colab-cli-env/bin/activate

SESSION=jepa-exp7-train
LOCAL_ROOT=/mnt/d/projects/JEPA
export WANDB_API_KEY="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"

echo "=== Verifying session is reachable ==="
colab --auth=adc status -s "$SESSION"

echo "=== 3. Creating remote directories ==="
cat << 'PYEOF' | colab --auth=adc exec -s "$SESSION" --timeout 60
import os
for d in ['/content/experiments/exp7_hybrid_decision', '/content/src', '/content/scripts', '/content/logs',
          '/content/data/intent_corpus', '/content/data/qqp_paraphrase_pairs', '/content/data/mcq_corpus',
          '/content/data/exp7_bool_corpus', '/content/data/exp7_score_corpus',
          '/content/data/exp7_diversity_corpus', '/content/data/exp7_typed_decisions']:
    os.makedirs(d, exist_ok=True)
print('dirs ready')
PYEOF

echo "=== 4. Uploading code ==="
for f in model.py data.py train.py eval.py calibrate.py smoke_test.py; do
  colab --auth=adc upload -s "$SESSION" "$LOCAL_ROOT/experiments/exp7_hybrid_decision/$f" "/content/experiments/exp7_hybrid_decision/$f"
done
colab --auth=adc upload -s "$SESSION" "$LOCAL_ROOT/src/local_data.py" /content/src/local_data.py
colab --auth=adc upload -s "$SESSION" "$LOCAL_ROOT/src/dataset_v7.py" /content/src/dataset_v7.py
colab --auth=adc upload -s "$SESSION" "$LOCAL_ROOT/scripts/build_exp7_data.py" /content/scripts/build_exp7_data.py

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

echo "=== 7. Restoring mcq/qqp/exp7-corpora from Drive backup, building on-VM if missing ==="
cat << 'PYEOF' | colab --auth=adc exec -s "$SESSION" --timeout 600
import shutil, subprocess, os
backup_root = '/content/drive/MyDrive/jepa_checkpoints/data_backup'

def restore_or_none(name):
    src = os.path.join(backup_root, name)
    dst = f'/content/data/{name}'
    if os.path.exists(src):
        shutil.copytree(src, dst, dirs_exist_ok=True)
        print(f'{name}: restored from Drive backup')
        return True
    return False

for name in ['mcq_corpus', 'qqp_paraphrase_pairs']:
    restore_or_none(name)

need_build = [n for n in ['exp7_bool_corpus', 'exp7_score_corpus', 'exp7_diversity_corpus',
                           'exp7_typed_decisions'] if not restore_or_none(n)]
if need_build:
    print(f'building on-VM: {need_build}')
    r = subprocess.run(['python3', '/content/scripts/build_exp7_data.py'], cwd='/content',
                        capture_output=True, text=True)
    print(r.stdout[-4000:]); print(r.stderr[-2000:])
    for name in need_build:
        src = f'/content/data/{name}'
        if os.path.exists(src):
            shutil.copytree(src, os.path.join(backup_root, name), dirs_exist_ok=True)
            print(f'{name}: built and backed up to Drive')
PYEOF

echo "=== 8. Logging into W&B on the VM ==="
cat << PYEOF | colab --auth=adc exec -s "$SESSION" --timeout 45
import subprocess
subprocess.run(['wandb', 'login', '$WANDB_API_KEY'], check=True)
print('wandb logged in')
PYEOF

echo "=== 9. Launching training (auto-resumes from exp7_latest.pt if present) ==="
cat << PYEOF | colab --auth=adc exec -s "$SESSION" --timeout 60
import subprocess, os
ckpt_dir = '/content/drive/MyDrive/jepa_checkpoints/exp7'
os.makedirs(ckpt_dir, exist_ok=True)
log = open(os.path.join(ckpt_dir, 'train_full.log'), 'w')
env = os.environ.copy()
env['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
env['WANDB_API_KEY'] = '$WANDB_API_KEY'
resume_path = os.path.join(ckpt_dir, 'exp7_latest.pt')
args = ['python3', '/content/experiments/exp7_hybrid_decision/train.py',
     '--k_max', '1', '--epochs', '20', '--steps_per_epoch', '500',
     '--batch_size', '8', '--grad_accum', '4', '--option_chunk_size', '64', '--patience', '6',
     '--wandb_project', 'jepa-exp7', '--wandb_run_id', 'exp7a', '--ckpt_dir', ckpt_dir]
if os.path.exists(resume_path):
    args += ['--resume', resume_path]
    print('resuming from', resume_path)
else:
    print('no checkpoint found -- starting fresh')
proc = subprocess.Popen(args, cwd='/content/experiments/exp7_hybrid_decision',
    stdout=log, stderr=subprocess.STDOUT, start_new_session=True, env=env)
with open('/content/logs/train_full.pid', 'w') as f:
    f.write(str(proc.pid))
print('launched, pid', proc.pid)
PYEOF

echo "=== Done. Training is running. Check progress at https://wandb.ai (project: jepa-exp7) ==="
echo "=== Keep this terminal window open, or run colab_exp7_watch.sh separately. ==="
