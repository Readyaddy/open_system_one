#!/bin/bash
# Resumes the setup from step 3 onward -- use this when `colab new` and
# `colab drivemount` already succeeded (e.g. the previous script hit a
# transient connection blip partway through) so you don't redo those
# steps or risk creating a duplicate session.

set -e
source ~/colab-cli-env/bin/activate

SESSION=jepa-train
LOCAL_ROOT=/mnt/d/projects/JEPA
export WANDB_API_KEY="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"

echo "=== Verifying session is reachable ==="
colab --auth=adc status -s "$SESSION"

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

echo "=== 8. Launching training (resumes from epoch 7: val 94.0%, zero_shot 44.5%) ==="
cat << PYEOF | colab --auth=adc exec -s "$SESSION" --timeout 60
import subprocess, os
ckpt_dir = '/content/drive/MyDrive/jepa_checkpoints/exp6'
log = open(os.path.join(ckpt_dir, 'train_full.log'), 'w')
env = os.environ.copy()
env['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
env['WANDB_API_KEY'] = '$WANDB_API_KEY'
proc = subprocess.Popen(
    ['python3', '/content/experiments/exp6_diverse_data_qqp_aux/train.py',
     '--epochs', '30', '--batch_size', '384', '--qqp_batch_size', '384', '--mcq_batch_size', '224',
     '--grad_accum', '1', '--outcome_chunk_size', '255',
     '--qqp_every_n_steps', '1', '--mcq_every_n_steps', '1', '--patience', '8',
     '--intent_options', '50',
     '--wandb_project', 'jepa-exp6', '--wandb_run_id', 'exp6',
     '--resume', '/content/drive/MyDrive/jepa_checkpoints/exp6/exp6_latest.pt',
     '--ckpt_dir', ckpt_dir],
    cwd='/content/experiments/exp6_diverse_data_qqp_aux',
    stdout=log, stderr=subprocess.STDOUT, start_new_session=True, env=env)
with open('/content/logs/train_full.pid', 'w') as f:
    f.write(str(proc.pid))
print('launched, pid', proc.pid)
PYEOF

echo "=== Done. Training is running. Check progress at https://wandb.ai (project: jepa-exp6) ==="
echo "=== Keep this terminal window open. ==="
