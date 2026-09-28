# Waits for exp8a3 to exit, checks it finished cleanly, then launches exp8b.
# exp8b = "learn yes/no from the content, not the slot": continues from exp8a3's
# best.pt with bool upweighted and 70% of bool examples rendered as descriptive
# criteria (Jev's `noul` format); adds bool/<src>@desc and typed/bool@swapped
# val groups. Other families stay in the mix so nothing is forgotten.
$ErrorActionPreference = "Stop"
$root = "D:\projects\JEPA"
$a3 = "$root\checkpoints\exp8_s1_rlcd\exp8a3"
$b = "$root\checkpoints\exp8_s1_rlcd\exp8b"
New-Item -ItemType Directory -Force $b | Out-Null
$log = "$b\queue.log"
function Log($m) { "$(Get-Date -Format s) $m" | Tee-Object -FilePath $log -Append }

Log "waiting for exp8a3 to exit"
while (Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -like '*train.py*exp8a3*' }) {
    Start-Sleep -Seconds 30
}
$tail = Get-Content "$a3\train.log" -Tail 5 -Raw
if ($tail -notmatch "done\.|early stop") {
    Log "exp8a3 did NOT finish cleanly (no 'done.'/'early stop' in train.log) -- not launching exp8b"
    exit 1
}
Log "exp8a3 finished cleanly; best.pt -> exp8b init"
Start-Sleep -Seconds 20   # let wandb/workers release the GPU

$trainArgs = @("train.py", "--run_name", "exp8b",
          "--init_ckpt", "$a3\best.pt",
          "--with_bool_desc", "--p_bool_desc", "0.7",
          "--w_bool", "0.40", "--w_intent", "0.15", "--w_mcq", "0.15", "--w_score", "0.10", "--w_diversity", "0.20",
          "--base_lr", "5e-6", "--warmup_frac", "0.05", "--max_steps", "2000",
          "--eval_every", "250", "--patience", "4")
$p = Start-Process -FilePath "C:\Users\Addy\miniforge3\python.exe" -ArgumentList $trainArgs `
    -WorkingDirectory "$root\experiments\exp8_s1_rlcd" `
    -RedirectStandardOutput "$b\stdout.log" -RedirectStandardError "$b\stderr.log" -WindowStyle Hidden -PassThru
Log "launched exp8b PID $($p.Id): $($trainArgs -join ' ')"

