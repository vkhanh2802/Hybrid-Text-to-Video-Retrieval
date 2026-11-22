param(
    [string]$Python = "",
    [string]$WorkerConfig = "config/school-runtime-siglip2.local.json",
    [ValidateSet("gemini", "groq")][string]$Provider = "gemini",
    [string]$Model = "",
    [switch]$WithQwen
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
# Manual DRES picking: Qwen is off by default (no model load, no image reads for
# scoring). Pass -WithQwen to restore the old 20-frame rerank.
if ($WithQwen) { $env:KIS_DISABLE_VLM = "0" } else { $env:KIS_DISABLE_VLM = "1" }
$settingsPath = Join-Path $ProjectRoot $WorkerConfig
if (-not (Test-Path -LiteralPath $settingsPath)) {
    throw "School worker settings missing. Deploy the worker first."
}
$settings = Get-Content -LiteralPath $settingsPath -Raw | ConvertFrom-Json
$env:KIS_WORKER_URL = $settings.url
$env:KIS_WORKER_TOKEN = $settings.token
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
if ($null -ne $settings.vlm_batch_size) {
    $batchSize = [int]$settings.vlm_batch_size
    if ($batchSize -lt 1 -or $batchSize -gt 20) { throw "Qwen batch size must be in [1,20]." }
    $env:KIS_VLM_BATCH_SIZE = [string]$batchSize
}
if ($null -ne $settings.ocr_format) {
    $env:KIS_OCR_FORMAT = [string]$settings.ocr_format
    $env:KIS_OCR_MODE = [string]$settings.ocr_mode
    $env:KIS_OCR_ROOT = [string]$settings.ocr_root
    $env:KIS_OCR_MANIFEST_SHA256 = [string]$settings.ocr_manifest_sha256
    $env:KIS_OCR_WEIGHT = ([double]$settings.ocr_weight).ToString([System.Globalization.CultureInfo]::InvariantCulture)
}
$tunnel = $null
try {
    $listener = Get-NetTCPConnection -LocalPort $settings.local_port -State Listen -ErrorAction SilentlyContinue
    if (-not $listener) {
        $forward = "127.0.0.1:$($settings.local_port):127.0.0.1:$($settings.remote_port)"
        $tunnel = Start-Process -FilePath "ssh.exe" -ArgumentList '-N','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','ExitOnForwardFailure=yes','-o','ServerAliveInterval=30','-o','ServerAliveCountMax=3','-L',$forward,$settings.target -PassThru -WindowStyle Hidden
    }
    # One-command start: if the worker is down, start it in its existing root (same token,
    # same settings) and wait until it has loaded. The token never leaves the school host.
    function Test-SchoolWorker {
        try {
            $reply = Invoke-WebRequest -Uri "$($settings.url)/status" -Headers @{ Authorization = "Bearer $($settings.token)" } -UseBasicParsing -TimeoutSec 5
            return $reply.StatusCode -eq 200
        } catch { return $false }
    }
    if (-not (Test-SchoolWorker)) {
        $port = [int]$settings.remote_port
        $workerEnv = 'PYTHONPATH=$R/source CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false FAISS_OPT_LEVEL=generic'
        if ($null -ne $settings.ocr_format) {
            $weight = ([double]$settings.ocr_weight).ToString([System.Globalization.CultureInfo]::InvariantCulture)
            $workerEnv += " KIS_OCR_FORMAT=$($settings.ocr_format) KIS_OCR_MODE=$($settings.ocr_mode) KIS_OCR_ROOT=$($settings.ocr_root) KIS_OCR_MANIFEST_SHA256=$($settings.ocr_manifest_sha256) KIS_OCR_WEIGHT=$weight"
        }
        if ($null -ne $settings.video_groups) { $workerEnv += " KIS_VIDEO_GROUPS=$($settings.video_groups -join ',')" }
        # No double quotes: Windows PowerShell 5.1 mangles them in native arguments.
        # The pgrep pattern is anchored to the python command, so it cannot match this bash -c line.
        # Only the worker goes to the background: a backgrounded 'cd && env' subshell would keep ssh open.
        $remote = ('R={0}; if pgrep -u $(id -un) -f ''^[^ ]*python[^ ]* -u -m kis[.]school_worker .*--port {1}( |$)'' >/dev/null; then echo loading; ' +
                   'else cd $R/source || exit 1; env {2} setsid nohup $HOME/KISBranche/.venv/bin/python -u -m kis.school_worker ' +
                   '--dataset {3} --token-file $R/worker.token --port {1} >> $R/worker.log 2>&1 < /dev/null & echo started; fi') -f $settings.remote_root, $port, $workerEnv, $settings.dataset
        $state = & ssh.exe -o BatchMode=yes -o StrictHostKeyChecking=yes -o ConnectTimeout=15 $settings.target $remote
        if ($LASTEXITCODE -ne 0) { throw "Cannot reach the school host over SSH; check VPN." }
        Write-Host "School worker $state on port $port; waiting for it to load (about 90 s)..."
        $deadline = (Get-Date).AddSeconds(300)
        while (-not (Test-SchoolWorker)) {
            if ((Get-Date) -gt $deadline) {
                throw "School worker not ready after 5 min; read $($settings.remote_root)/worker.log on the school host."
            }
            Start-Sleep -Seconds 5
        }
        Write-Host "School worker ready."
    }
    # start_kis only calls the query planner API and local web. It inherits the worker connection;
    # no planner/Drive/DRES credentials are sent to the school host.
    & (Join-Path $ProjectRoot "scripts/start_kis.ps1") -Python $Python -Provider $Provider -Model $Model
} finally {
    if ($null -ne $tunnel -and -not $tunnel.HasExited) {
        Stop-Process -Id $tunnel.Id -ErrorAction SilentlyContinue
    }
    Remove-Item Env:KIS_WORKER_TOKEN -ErrorAction SilentlyContinue
    Remove-Item Env:KIS_WORKER_URL -ErrorAction SilentlyContinue
}
