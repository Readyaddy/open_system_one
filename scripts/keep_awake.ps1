# Holds "system required + display required" while a training process runs, so
# Modern Standby can't start on idle timeout (it begins when the display turns
# off, which ES_SYSTEM_REQUIRED alone doesn't prevent). Nothing in the power
# plan is changed; the request is released when this script exits.
#   powershell -File keep_awake.ps1 -Match exp8a3
param([string]$Match = "train.py")
Add-Type -Namespace Win32 -Name Power -MemberDefinition @"
[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);
"@
$ES_CONTINUOUS = [uint32]"0x80000000"; $ES_SYSTEM = [uint32]1; $ES_DISPLAY = [uint32]2
[Win32.Power]::SetThreadExecutionState($ES_CONTINUOUS -bor $ES_SYSTEM -bor $ES_DISPLAY) | Out-Null
Write-Output "$(Get-Date -Format s) holding awake while '*$Match*' runs"
while ($true) {
    $alive = Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -like "*$Match*" }
    if (-not $alive) { break }
    # Re-assert periodically; also nudges the idle timer on systems that ignore the continuous flag.
    [Win32.Power]::SetThreadExecutionState($ES_CONTINUOUS -bor $ES_SYSTEM -bor $ES_DISPLAY) | Out-Null
    Start-Sleep -Seconds 30
}
[Win32.Power]::SetThreadExecutionState($ES_CONTINUOUS) | Out-Null
Write-Output "$(Get-Date -Format s) training process gone; released"
