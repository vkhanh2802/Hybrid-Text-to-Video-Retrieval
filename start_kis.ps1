param([string]$Python = "")

$ErrorActionPreference = "Stop"
if (-not $Python) {
    $condaPython = Join-Path $env:USERPROFILE "miniconda3\envs\focusguard_ai\python.exe"
    if (Test-Path -LiteralPath $condaPython) {
        $Python = $condaPython
    } else {
        $Python = (Get-Command python -ErrorAction Stop).Source
    }
}

$env:OMP_NUM_THREADS = "4"
$env:MKL_NUM_THREADS = "4"
$env:PYTHONIOENCODING = "utf-8"
Push-Location -LiteralPath $PSScriptRoot
try {
    & $Python -m kis.web_app
} finally {
    Pop-Location
}
exit $LASTEXITCODE
