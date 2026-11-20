param([string]$Python = "", [ValidateSet("gemini", "groq")][string]$Provider = "gemini", [string]$Model = "", [switch]$CheckApi, [switch]$PromptForKey)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
if (-not $Python) {
    $condaPython = Join-Path $env:USERPROFILE "miniconda3\envs\focusguard_ai\python.exe"
    if (Test-Path -LiteralPath $condaPython) {
        $Python = $condaPython
    } else {
        $Python = (Get-Command python -ErrorAction Stop).Source
    }
}

$preset = @{
    gemini = @{ KeyName = "GEMINI_API_KEY"; Protocol = "gemini"; BaseUrl = "https://generativelanguage.googleapis.com"; Model = "gemini-flash-lite-latest"; Timeout = "15" }
    groq   = @{ KeyName = "GROQ_API_KEY"; Protocol = "openai"; BaseUrl = "https://api.groq.com/openai/v1"; Model = "openai/gpt-oss-20b"; Timeout = "60" }
}[$Provider]
if (-not $Model) { $Model = $preset.Model }
$keyName = $preset.KeyName
$key = [Environment]::GetEnvironmentVariable($keyName, "Process")
if ($PromptForKey) {
    $key = ""
} elseif ([string]::IsNullOrWhiteSpace($key)) {
    # Read User scope explicitly: terminals spawned by an already-open app can
    # inherit a stale environment even after the Windows setting was updated.
    $key = [Environment]::GetEnvironmentVariable($keyName, "User")
}
if ([string]::IsNullOrWhiteSpace($key)) {
    $secureKey = Read-Host "Nhap $keyName (an ky tu)" -AsSecureString
    try {
        $credential = [System.Management.Automation.PSCredential]::new($Provider, $secureKey)
        $key = $credential.GetNetworkCredential().Password
    } finally {
        $secureKey.Dispose()
    }
    if ([string]::IsNullOrWhiteSpace($key)) {
        throw "$keyName must not be empty"
    }
}
$env:KIS_LLM_PROVIDER = $Provider
$env:KIS_LLM_PROTOCOL = $preset.Protocol
$env:KIS_LLM_MODEL = $Model
$env:KIS_LLM_BASE_URL = $preset.BaseUrl
$env:KIS_LLM_RESPONSE_FORMAT = "json_schema"
# Short Gemini timeout: a stalled call hands over to the OpenCode fallback sooner.
$env:KIS_LLM_TIMEOUT_SECONDS = $preset.Timeout
$env:KIS_LLM_API_KEY = $key.Trim()
# Child tools that read the provider-named variable (e.g. GROQ_API_KEY) see the same key.
[Environment]::SetEnvironmentVariable($keyName, $key.Trim(), "Process")
Remove-Variable key -ErrorAction SilentlyContinue

$env:OMP_NUM_THREADS = "4"
$env:MKL_NUM_THREADS = "4"
$env:PYTHONIOENCODING = "utf-8"
Push-Location -LiteralPath $ProjectRoot
try {
    if ($CheckApi) {
        & $Python -m kis.structure_query --no-save
    } else {
        & $Python -m kis.web_app
    }
} finally {
    Pop-Location
}
exit $LASTEXITCODE
