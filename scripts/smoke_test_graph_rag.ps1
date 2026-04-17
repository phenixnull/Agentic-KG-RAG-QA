param(
    [string]$ApiBase = "http://127.0.0.1:5150",
    [string]$AdminUser = "admin",
    [string]$AdminPassword = "123123",
    [string]$SmokeFile = "",
    [int]$TimeoutSeconds = 900
)

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($SmokeFile)) {
    $SmokeFile = Join-Path $projectRoot "saves\smoke\smoke_graph_test.md"
}

function Wait-Health {
    param(
        [string]$Url,
        [int]$Timeout = 300
    )

    $deadline = (Get-Date).AddSeconds($Timeout)
    while ((Get-Date) -lt $deadline) {
        try {
            $response = Invoke-RestMethod -Method Get -Uri $Url -TimeoutSec 10
            if ($response.status -eq "ok") {
                return
            }
        }
        catch {
            Start-Sleep -Seconds 5
        }
    }

    throw "Timed out waiting for health endpoint: $Url"
}

function Invoke-ApiJson {
    param(
        [string]$Method,
        [string]$Uri,
        [object]$Body = $null,
        [hashtable]$Headers = @{}
    )

    $params = @{
        Method     = $Method
        Uri        = $Uri
        TimeoutSec = 60
        Headers    = $Headers
    }

    if ($null -ne $Body) {
        $params.ContentType = "application/json"
        $params.Body = ($Body | ConvertTo-Json -Depth 10)
    }

    return Invoke-RestMethod @params
}

Write-Host "Waiting for API health..." -ForegroundColor Cyan
Wait-Health -Url "$ApiBase/api/system/health" -Timeout 600

$firstRun = Invoke-RestMethod -Method Get -Uri "$ApiBase/api/auth/check-first-run" -TimeoutSec 30
if ($firstRun.first_run -eq $true) {
    Write-Host "Initializing admin account..." -ForegroundColor Yellow
    $null = Invoke-ApiJson -Method Post -Uri "$ApiBase/api/auth/initialize" -Body @{
        user_id  = $AdminUser
        password = $AdminPassword
    }
}

Write-Host "Logging in as admin..." -ForegroundColor Cyan
$tokenResponse = Invoke-RestMethod `
    -Method Post `
    -Uri "$ApiBase/api/auth/token" `
    -Body @{ username = $AdminUser; password = $AdminPassword } `
    -ContentType "application/x-www-form-urlencoded" `
    -TimeoutSec 30

$token = $tokenResponse.access_token
if ([string]::IsNullOrWhiteSpace($token)) {
    throw "Failed to acquire admin token."
}

$authHeaders = @{ Authorization = "Bearer $token" }
$dbName = "smoke_lightrag_{0}" -f ([guid]::NewGuid().ToString("N").Substring(0, 8))

Write-Host "Creating lightrag knowledge base $dbName ..." -ForegroundColor Cyan
$database = Invoke-ApiJson -Method Post -Uri "$ApiBase/api/knowledge/databases" -Headers $authHeaders -Body @{
    database_name    = $dbName
    description      = "Standalone graph+RAG smoke test"
    embed_model_name = "local/bge-m3"
    kb_type          = "lightrag"
    llm_info         = @{
        model_spec = "fucheers/gemini-3-flash"
    }
    additional_params = @{}
}

$dbId = $database.db_id
if ([string]::IsNullOrWhiteSpace($dbId)) {
    throw "Knowledge base creation did not return db_id."
}

Write-Host "Uploading smoke markdown file..." -ForegroundColor Cyan
$uploadRaw = & curl.exe -sS -X POST "$ApiBase/api/knowledge/files/upload?db_id=$dbId" `
    -H "Authorization: Bearer $token" `
    -F "file=@$SmokeFile"

if ($LASTEXITCODE -ne 0) {
    throw "curl upload failed."
}

$upload = $uploadRaw | ConvertFrom-Json
$filePath = $upload.file_path
if ([string]::IsNullOrWhiteSpace($filePath)) {
    throw "Upload did not return file_path."
}
$contentHash = $upload.content_hash

Write-Host "Submitting ingest task..." -ForegroundColor Cyan
$ingest = Invoke-ApiJson -Method Post -Uri "$ApiBase/api/knowledge/databases/$dbId/documents" -Headers $authHeaders -Body @{
    items  = @($filePath)
    params = @{
        content_type  = "file"
        auto_index    = $true
        chunk_size    = 600
        chunk_overlap = 120
        content_hashes = @{
            $filePath = $contentHash
        }
    }
}

$taskId = $ingest.task_id
if ([string]::IsNullOrWhiteSpace($taskId)) {
    throw "Ingest task submission did not return task_id."
}

Write-Host "Polling task $taskId ..." -ForegroundColor Cyan
$deadline = (Get-Date).AddSeconds($TimeoutSeconds)
$task = $null
while ((Get-Date) -lt $deadline) {
    $taskResponse = Invoke-RestMethod -Method Get -Uri "$ApiBase/api/tasks/$taskId" -Headers $authHeaders -TimeoutSec 30
    $task = $taskResponse.task
    if ($task.status -in @("success", "failed", "cancelled")) {
        break
    }
    Start-Sleep -Seconds 5
}

if ($null -eq $task) {
    throw "Task polling failed."
}
if ($task.status -ne "success") {
    throw "Ingest task ended with status $($task.status): $($task.message)"
}
if (($task.result.failed | ForEach-Object { [int]$_ }) -gt 0) {
    throw "Ingest task reported failed items: $($task.result | ConvertTo-Json -Depth 10)"
}

Write-Host "Running knowledge query smoke test..." -ForegroundColor Cyan
$queryResponse = Invoke-ApiJson -Method Post -Uri "$ApiBase/api/knowledge/databases/$dbId/query-test" -Headers $authHeaders -Body @{
    query = "What is the first maintenance action when the suction filter is blocked?"
    meta  = @{}
}

$queryText = $queryResponse | ConvertTo-Json -Depth 10
if ($queryText -notmatch "clean the suction filter") {
    throw "Smoke query did not return the expected maintenance answer."
}

Write-Host "Checking graph stats..." -ForegroundColor Cyan
$graphStats = Invoke-RestMethod -Method Get -Uri "$ApiBase/api/graph/stats?db_id=$dbId" -Headers $authHeaders -TimeoutSec 30
if ($graphStats.success -ne $true) {
    throw "Graph stats request failed."
}
if (($graphStats.data.total_nodes | ForEach-Object { [int]$_ }) -le 0) {
    throw "Graph stats did not report any nodes."
}

Write-Host ""
Write-Host "Smoke test passed." -ForegroundColor Green
Write-Host "  db_id: $dbId"
Write-Host "  task_id: $taskId"
Write-Host "  query response: $queryText"
Write-Host "  graph nodes: $($graphStats.data.total_nodes)"
