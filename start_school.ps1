# Entry point: .\start_school.ps1 starts the school worker if needed, then the web.
# Python is optional; scripts/start_kis.ps1 finds the focusguard_ai conda env first.
param([string]$Python = "", [string]$WorkerConfig = "config/school-runtime-siglip2.local.json",
      [ValidateSet("gemini", "groq")][string]$Provider = "gemini", [string]$Model = "", [switch]$WithQwen)

& (Join-Path $PSScriptRoot "scripts/legacy/start_school.ps1") @PSBoundParameters
exit $LASTEXITCODE
