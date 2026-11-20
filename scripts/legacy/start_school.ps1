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
