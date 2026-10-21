# run_jitter.ps1 - prompt-jitter retraining of the mixed-prompt model, then its clean and noisy-prompt evaluation.
# Run in a PowerShell window with the training environment active:
#   Set-ExecutionPolicy -Scope Process Bypass -Force; .\run_jitter.ps1
# About 16-17 h in total. Everything is resumable: if it stops, run the same file again.
$ErrorActionPreference = "Continue"   # warnings on stderr must not stop the chain
Set-Location $PSScriptRoot
$log = "run_jitter_log.txt"
function Step($name, [scriptblock]$cmd) {
    "`n===== $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  $name" | Tee-Object -FilePath $log -Append
    & $cmd 2>&1 | Tee-Object -FilePath $log -Append
    if ($LASTEXITCODE -ne 0) { "!! $name failed (exit $LASTEXITCODE) - stopping. See run_jitter_log.txt" | Tee-Object -FilePath $log -Append; exit 1 }
}

# 0. check that this window's python is the one with SAM and CUDA
"python used: $((Get-Command python).Source)" | Tee-Object -FilePath $log -Append
python -c "import segment_anything, torch; assert torch.cuda.is_available(), 'no CUDA'; print('segment_anything OK, GPU', torch.cuda.get_device_name(0))" 2>&1 | Tee-Object -FilePath $log -Append
if ($LASTEXITCODE -ne 0) { "!! this python has no segment_anything/CUDA. Activate the environment you train with, then run .\run_jitter.ps1 again" | Tee-Object -FilePath $log -Append; exit 1 }

# 1. smoke test (10-15 min): catches errors before the long run
if (-not (Test-Path "lora_v4_jit\tile1024_smoke\history.json")) {
    Step "smoke test" { python lora_train_v4_jit.py --mode tile1024 --prompt-jitter 0.5 --jitter-type uniform --out-name lora_v4_jit --smoke }
}

# 2. full training (about 9 h), same seed, schedule and selection rule as the paper's v4 run
$extra = @(); if (Test-Path "lora_v4_jit\tile1024\history.json") { $extra = @("--resume") }
if (-not (Test-Path "lora_v4_jit\tile1024\best.pt")) {
    Step "training" { python lora_train_v4_jit.py --mode tile1024 --prompt-jitter 0.5 --jitter-type uniform --out-name lora_v4_jit @extra }
}

# 3. clean evaluation on iSAID val, 9 prompt types (about 4 h)
Step "clean evaluation" { python sam_eval_v4.py --model sam1 --mode tile1024 --lora lora_v4_jit\tile1024\best.pt --tag v4jit }

# 4. noisy-prompt evaluation, same draws as the earlier noise runs (about 3 h)
Step "noise evaluation" { python sam_eval_v4.py --model sam1 --mode tile1024 --lora lora_v4_jit\tile1024\best.pt --tag v4jit --strategies BBN5,BBN10,BBN20,CPJ5,CPJ10,CPJ20 --out results_v4_noise }

"`n===== $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  ALL DONE" | Tee-Object -FilePath $log -Append
