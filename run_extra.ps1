# run_extra.ps1 - the three extra experiments, one after the other (about 45-50 h of GPU time in total):
#   A. shifted-prompt adapter on NWPU VHR-10                                   (~30 min)
#   B. detector boxes: YOLO11-m trained on iSAID train -> boxes on iSAID val -> SAM with these boxes
#      for frozen SAM, the exact-prompt adapter and the shifted-prompt adapter  (~7-9 h)
#   C. shifted-prompt adapter, seeds 1 and 2: training + clean + shifted-prompt evaluation (~17 h each)
# Run in a PowerShell window with the training environment active:   Set-ExecutionPolicy -Scope Process Bypass -Force; .\run_extra.ps1
# Resumable: run the same line again after any stop; finished steps are skipped.
# If a step of block A or B fails, the script logs it and goes on with the next block.
$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot
$log = "run_extra_log.txt"
function Say($s) { $s | Tee-Object -FilePath $log -Append | Out-Host }
function Step($name, [scriptblock]$cmd) {
    Say "`n===== $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  $name"
    & $cmd 2>&1 | Tee-Object -FilePath $log -Append | Out-Host
    if ($LASTEXITCODE -ne 0) { Say "!! $name failed (exit $LASTEXITCODE)"; return $false }
    return $true
}
function Done($folder) { return (Test-Path (Join-Path $folder "log.txt")) -and (Select-String -Path (Join-Path $folder "log.txt") -Pattern "== finished" -Quiet) }

Say "python used: $((Get-Command python).Source)"
python -c "import segment_anything, torch; assert torch.cuda.is_available(); print('segment_anything OK, GPU', torch.cuda.get_device_name(0))" 2>&1 | Tee-Object -FilePath $log -Append
if ($LASTEXITCODE -ne 0) { Say "!! wrong python: activate the training environment and run again"; exit 1 }

# ---------------- A. NWPU VHR-10, shifted-prompt adapter
if (-not (Done "results_ext_nwpu\sam1lora-v4jit_tile1024")) {
    $env:EVAL_ANN = "D:\NWPU\annotations.json"; $env:EVAL_IMG_DIR = "D:\NWPU\NWPU VHR-10 dataset\positive image set"
    $null = Step "A. NWPU evaluation, shifted-prompt adapter" { python sam_eval_v4.py --model sam1 --mode tile1024 --lora lora_v4_jit\tile1024\best.pt --tag v4jit --out results_ext_nwpu }
    Remove-Item Env:EVAL_ANN, Env:EVAL_IMG_DIR -ErrorAction SilentlyContinue
}

# ---------------- B. detector boxes
$okB = $true
python -c "import ultralytics" 2>$null
if ($LASTEXITCODE -ne 0) { $okB = Step "B0. install ultralytics" { python -m pip install "ultralytics==8.3.40" } }
if ($okB -and -not (Test-Path "D:\iSAID_Data\yolo_isaid\data.yaml")) { $okB = Step "B1. detection dataset from iSAID train" { python det_prepare.py } }
if ($okB -and -not (Test-Path "det\yolo11m_isaid\TRAINING_DONE")) {
    if (Test-Path "det\yolo11m_isaid\weights\last.pt") { $okB = Step "B2. detector training (resume)" { python det_train.py --resume } }
    else { $okB = Step "B2. detector training" { python det_train.py } }
}
if ($okB -and -not (Test-Path "det\det_boxes.json")) { $okB = Step "B3. detector boxes on iSAID val" { python det_boxes.py } }
if ($okB) {
    $env:DET_BOXES = (Resolve-Path "det\det_boxes.json").Path
    if (-not (Done "results_det\sam1_tile1024")) { $null = Step "B4. SAM with detector boxes: frozen" { python sam_eval_v4.py --model sam1 --mode tile1024 --strategies DET --out results_det } }
    if (-not (Done "results_det\sam1lora-v4_tile1024")) { $null = Step "B5. SAM with detector boxes: exact-prompt adapter" { python sam_eval_v4.py --model sam1 --mode tile1024 --lora lora_v4\tile1024\best.pt --tag v4 --strategies DET --out results_det } }
    if (-not (Done "results_det\sam1lora-v4jit_tile1024")) { $null = Step "B6. SAM with detector boxes: shifted-prompt adapter" { python sam_eval_v4.py --model sam1 --mode tile1024 --lora lora_v4_jit\tile1024\best.pt --tag v4jit --strategies DET --out results_det } }
    Remove-Item Env:DET_BOXES -ErrorAction SilentlyContinue
}

# ---------------- C. seeds 1 and 2 of the shifted-prompt adapter
foreach ($s in 1, 2) {
    $out = "lora_v4_jit_s$s"; $tag = "v4jits$s"
    if (-not (Test-Path "$out\tile1024\best.pt")) {
        $extra = @(); if (Test-Path "$out\tile1024\history.json") { $extra = @("--resume") }
        if (-not (Step "C. training, seed $s" { python lora_train_v4_jit.py --mode tile1024 --prompt-jitter 0.5 --jitter-type uniform --run-seed $s --out-name $out @extra })) { continue }
    }
    if (-not (Done "results_v4\sam1lora-$($tag)_tile1024")) { $null = Step "C. clean evaluation, seed $s" { python sam_eval_v4.py --model sam1 --mode tile1024 --lora "$out\tile1024\best.pt" --tag $tag } }
    if (-not (Done "results_v4_noise\sam1lora-$($tag)_tile1024")) { $null = Step "C. shifted-prompt evaluation, seed $s" { python sam_eval_v4.py --model sam1 --mode tile1024 --lora "$out\tile1024\best.pt" --tag $tag --strategies BBN5,BBN10,BBN20,CPJ5,CPJ10,CPJ20 --out results_v4_noise } }
}
Say "`n===== $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  ALL DONE (check the log for any '!!' lines)"
