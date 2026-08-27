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
        Set-Content -LiteralPath $PidFile -Value $managedProcess.Id -Encoding ascii
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

