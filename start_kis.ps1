# Compatibility entry point. The maintained launcher lives under scripts/.
param([string]$Python = "", [ValidateSet("gemini", "groq")][string]$Provider = "gemini", [string]$Model = "", [switch]$CheckApi, [switch]$PromptForKey)

& (Join-Path $PSScriptRoot "scripts/start_kis.ps1") @PSBoundParameters
exit $LASTEXITCODE
