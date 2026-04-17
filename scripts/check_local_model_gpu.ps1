$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot

Push-Location $projectRoot
try {
    & docker compose -f docker-compose.yml -f docker-compose.gpu.yml --env-file .env exec -T api `
        python -c "import os, torch; print('device_env', os.getenv('YUXI_LOCAL_MODEL_DEVICE')); print('torch', torch.__version__); print('cuda_version', torch.version.cuda); print('cuda_available', torch.cuda.is_available()); print('device_count', torch.cuda.device_count()); print('current_device', torch.cuda.current_device() if torch.cuda.is_available() else 'cpu')"

    if ($LASTEXITCODE -ne 0) {
        throw "GPU verification failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}
