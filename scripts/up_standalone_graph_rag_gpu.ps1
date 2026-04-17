param(
    [switch]$NoBuild
)

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$services = @(
    "postgres",
    "redis",
    "minio",
    "milvus",
    "graph",
    "sandbox-provisioner",
    "api",
    "worker",
    "web"
)

Push-Location $projectRoot
try {
    $dockerArgs = @(
        "compose",
        "-f", "docker-compose.yml",
        "-f", "docker-compose.gpu.yml",
        "--env-file", ".env",
        "up", "-d"
    )
    if (-not $NoBuild) {
        $dockerArgs += "--build"
    }
    $dockerArgs += $services

    Write-Host "Starting standalone YUXI graph+RAG stack with GPU local models..." -ForegroundColor Cyan
    & docker @dockerArgs
    if ($LASTEXITCODE -ne 0) {
        throw "docker compose up failed with exit code $LASTEXITCODE"
    }

    Write-Host ""
    Write-Host "Expected endpoints:" -ForegroundColor Green
    Write-Host "  API:      http://127.0.0.1:5150"
    Write-Host "  Web:      http://127.0.0.1:5273"
    Write-Host "  MinIO:    http://127.0.0.1:9101"
    Write-Host "  Neo4j:    http://127.0.0.1:7574"
    Write-Host "  Milvus:   http://127.0.0.1:19531"
    Write-Host ""
    Write-Host "GPU verification:" -ForegroundColor Green
    Write-Host "  powershell -ExecutionPolicy Bypass -File .\\scripts\\check_local_model_gpu.ps1"
}
finally {
    Pop-Location
}
