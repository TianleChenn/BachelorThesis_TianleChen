$LocalModelHealthUrl = "http://127.0.0.1:8080/v1/models"
try {
    $response = Invoke-RestMethod -Uri $LocalModelHealthUrl -Method Get -TimeoutSec 2
    if ($null -eq $response) { throw "Empty response" }
    Write-Host "Local Ministral: READY"
    Write-Host "Endpoint: http://127.0.0.1:8080/v1"
    exit 0
}
catch {
    Write-Host "Local Ministral: NOT RUNNING"
    exit 1
}
