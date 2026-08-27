$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir
$PidFile = Join-Path $ProjectRoot "artifacts\local_model_server.pid"

if (-not (Test-Path -LiteralPath $PidFile)) {
    Write-Host "No managed Local Ministral server is currently running."
    exit 0
}

$pidText = (Get-Content -Raw -LiteralPath $PidFile).Trim()
$process = $null
if ($pidText -match '^\d+$') {
    $process = Get-Process -Id ([int]$pidText) -ErrorAction SilentlyContinue
}
if ($null -eq $process) {
    Remove-Item -LiteralPath $PidFile -Force
    Write-Host "No managed Local Ministral server is currently running."
    exit 0
}
if ($process.ProcessName -notmatch '^llama-server$') {
    Write-Warning "The managed PID now belongs to another process; it was not stopped."
    Remove-Item -LiteralPath $PidFile -Force
    exit 1
}

Stop-Process -Id $process.Id
Remove-Item -LiteralPath $PidFile -Force
Write-Host "Local Ministral server stopped."
