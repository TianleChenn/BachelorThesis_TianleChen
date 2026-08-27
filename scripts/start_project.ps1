$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir
Set-Location $ProjectRoot

Write-Host "============================================================"
Write-Host "Athlete LLM Project Startup"
Write-Host "============================================================"

& (Join-Path $ScriptDir "ensure_local_model.ps1")
if ($LASTEXITCODE -ne 0) {
    Write-Error "Local Ministral could not be started. Streamlit was not launched."
    exit $LASTEXITCODE
}

Write-Host "============================================================"
Write-Host "Starting Frontend"
Write-Host "============================================================"
python -m streamlit run frontend.py
exit $LASTEXITCODE
