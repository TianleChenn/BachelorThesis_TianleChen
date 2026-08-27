$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir
$PidFile = Join-Path $ProjectRoot "artifacts\local_model_server.pid"

function Remove-StalePidFile {
    if (Test-Path -LiteralPath $PidFile) {
        Remove-Item -LiteralPath $PidFile -Force
    }
}

function Get-LocalModelPortOwner {
    $listener = Get-NetTCPConnection -LocalAddress "127.0.0.1" -LocalPort 8080 `
        -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($null -eq $listener) { return $null }
    return Get-Process -Id $listener.OwningProcess -ErrorAction SilentlyContinue
}

function Stop-VerifiedLlamaServer([System.Diagnostics.Process]$Process) {
    if ($Process.ProcessName -notmatch '^llama-server$') {
        Write-Warning "Refusing to stop a non-llama-server process."
        return $false
    }
    Stop-Process -Id $Process.Id
    $deadline = [DateTime]::UtcNow.AddSeconds(5)
    $remaining = Get-Process -Id $Process.Id -ErrorAction SilentlyContinue
    while ($null -ne $remaining -and [DateTime]::UtcNow -lt $deadline) {
        Start-Sleep -Milliseconds 100
        $remaining = Get-Process -Id $Process.Id -ErrorAction SilentlyContinue
    }
    if ($null -ne $remaining) {
        Stop-Process -Id $Process.Id -Force
    }
    return $true
}

$managedProcess = $null
if (Test-Path -LiteralPath $PidFile) {
    $pidText = (Get-Content -Raw -LiteralPath $PidFile).Trim()
    if ($pidText -match '^\d+$') {
        $candidate = Get-Process -Id ([int]$pidText) -ErrorAction SilentlyContinue
        if ($null -ne $candidate -and $candidate.ProcessName -match '^llama-server$') {
            $managedProcess = $candidate
        }
    }
    if ($null -eq $managedProcess) {
        Write-Warning "Removing a missing, invalid, or unrelated stale Local Ministral PID file."
        Remove-StalePidFile
    }
}

if ($null -eq $managedProcess) {
    $portOwner = Get-LocalModelPortOwner
    if ($null -eq $portOwner) {
        Write-Host "No managed Local Ministral server is currently running."
        exit 0
    }
    if ($portOwner.ProcessName -notmatch '^llama-server$') {
        Write-Warning "Port 8080 is owned by a non-llama-server process; it was not stopped."
        exit 1
    }
    $managedProcess = $portOwner
    Write-Host "Recovered llama-server process $($managedProcess.Id) from port 8080."
}

if (-not (Stop-VerifiedLlamaServer $managedProcess)) {
    Remove-StalePidFile
    exit 1
}
Remove-StalePidFile
Write-Host "Local Ministral server stopped."
