[CmdletBinding()]
param(
    [int]$ReadyTimeoutSeconds = 600,
    [string]$Model = "mistralai/Ministral-3-8B-Instruct-2512-GGUF:Q4_K_M",
    [string]$ModelAlias = "Ministral-3-8B-Local"
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir
$HealthUrl = "http://127.0.0.1:8080/v1/models"
$BaseUrl = "http://127.0.0.1:8080/v1"
$ArtifactsDir = Join-Path $ProjectRoot "artifacts"
$PidFile = Join-Path $ArtifactsDir "local_model_server.pid"
$StdoutLog = Join-Path $ArtifactsDir "local_model_server_stdout.log"
$StderrLog = Join-Path $ArtifactsDir "local_model_server_stderr.log"

function Test-LocalModelReady {
    try {
        $response = Invoke-RestMethod -Uri $HealthUrl -Method Get -TimeoutSec 2
        return $null -ne $response
    }
    catch { return $false }
}

function Get-LocalModelPortOwner {
    $listener = Get-NetTCPConnection -LocalAddress "127.0.0.1" -LocalPort 8080 `
        -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($null -eq $listener) { return $null }
    return Get-Process -Id $listener.OwningProcess -ErrorAction SilentlyContinue
}

function Save-ManagedLocalModelPid([System.Diagnostics.Process]$Process) {
    New-Item -ItemType Directory -Path $ArtifactsDir -Force | Out-Null
    Set-Content -LiteralPath $PidFile -Value $Process.Id -Encoding ascii
}

function Get-ManagedLocalModelProcess {
    if (-not (Test-Path -LiteralPath $PidFile)) { return $null }
    $pidText = (Get-Content -Raw -LiteralPath $PidFile).Trim()
    if ($pidText -notmatch '^\d+$') {
        Remove-Item -LiteralPath $PidFile -Force
        return $null
    }
    $process = Get-Process -Id ([int]$pidText) -ErrorAction SilentlyContinue
    if ($null -eq $process -or $process.ProcessName -notmatch '^llama-server$') {
        Remove-Item -LiteralPath $PidFile -Force
        return $null
    }
    return $process
}

function Wait-ForLocalModel([System.Diagnostics.Process]$Process) {
    $deadline = [DateTime]::UtcNow.AddSeconds($ReadyTimeoutSeconds)
    Write-Host "Waiting for Local Ministral to become ready..."
    while ([DateTime]::UtcNow -lt $deadline) {
        if (Test-LocalModelReady) { return $true }
        if ($null -ne $Process) {
            $Process.Refresh()
            if ($Process.HasExited) {
                Write-Error "Local model server exited before readiness. Check $StderrLog"
                return $false
            }
        }
        Start-Sleep -Seconds 2
    }
    Write-Error "Local model did not become ready within $ReadyTimeoutSeconds seconds. Check $StderrLog"
    return $false
}

try {
    if (Test-LocalModelReady) {
        $readyProcess = Get-LocalModelPortOwner
        if ($null -eq $readyProcess) {
            Write-Error "Port 8080 is ready, but its listening process could not be identified."
            exit 1
        }
        if ($readyProcess.ProcessName -notmatch '^llama-server$') {
            Write-Error "Port 8080 is owned by a non-llama-server process; refusing to adopt it."
            exit 1
        }
        Save-ManagedLocalModelPid $readyProcess
        Write-Host "Recorded ready llama-server process $($readyProcess.Id)."
        Write-Host "Local Ministral: READY"
        Write-Host "Endpoint: $BaseUrl"
        exit 0
    }

    New-Item -ItemType Directory -Path $ArtifactsDir -Force | Out-Null
    $managedProcess = Get-ManagedLocalModelProcess
    if ($null -eq $managedProcess) {
        $llamaServerPath = $env:LLAMA_SERVER_PATH
        if ($llamaServerPath) {
            if (-not (Test-Path -LiteralPath $llamaServerPath -PathType Leaf)) {
                Write-Error "LLAMA_SERVER_PATH does not point to a file: $llamaServerPath"
                exit 1
            }
        }
        else {
            $llamaServer = Get-Command llama-server -ErrorAction SilentlyContinue
            if ($null -eq $llamaServer) {
                Write-Error "llama-server was not found. Add it to PATH or set LLAMA_SERVER_PATH."
                exit 1
            }
            $llamaServerPath = $llamaServer.Source
        }

        Write-Host "Starting Local Ministral-3-8B on localhost..."
        $arguments = @(
            "-hf", $Model,
            "--alias", $ModelAlias,
            "--host", "127.0.0.1",
            "--port", "8080"
        )
        $managedProcess = Start-Process -FilePath $llamaServerPath -ArgumentList $arguments `
            -RedirectStandardOutput $StdoutLog -RedirectStandardError $StderrLog `
            -WindowStyle Hidden -PassThru
        Save-ManagedLocalModelPid $managedProcess
    }
    else {
        Write-Host "Reusing managed llama-server process $($managedProcess.Id)."
    }

    if (-not (Wait-ForLocalModel $managedProcess)) { exit 1 }
    Write-Host "Local Ministral: READY"
    Write-Host "Endpoint: $BaseUrl"
    exit 0
}
catch {
    Write-Error "Local Ministral startup failed: $($_.Exception.Message)"
    exit 1
}
