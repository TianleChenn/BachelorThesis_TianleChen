[CmdletBinding()]
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$AnalysisArguments
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir

& (Join-Path $ScriptDir "ensure_local_model.ps1")
if ($LASTEXITCODE -ne 0) {
    Write-Error "Real Local Edge verification cannot continue because Local Ministral is unavailable. No Cloud fallback was attempted."
    exit $LASTEXITCODE
}

Push-Location $ProjectRoot
try {
    & python (Join-Path $ScriptDir "verify_local_edge_analyses.py") @AnalysisArguments
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}

