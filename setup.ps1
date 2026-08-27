<#
.SYNOPSIS
  One-shot setup for prod-rag: installs deps (CPU or GPU torch), starts
  Qdrant + Postgres containers, optionally ingests, then serves the API + UI.

.EXAMPLE
  .\setup.ps1                 # auto-detect GPU, setup + serve
  .\setup.ps1 -Device gpu     # force CUDA install
  .\setup.ps1 -Device cpu -Ingest   # CPU install, run ingestion before serving
#>
param(
    [ValidateSet("auto", "gpu", "cpu")] [string]$Device = "auto",
    [switch]$Ingest,
    [switch]$NoServe
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

function Fail($msg) { Write-Host "ERROR: $msg" -ForegroundColor Red; exit 1 }
function Step($msg) { Write-Host "`n==> $msg" -ForegroundColor Cyan }

# ---- prerequisites -----------------------------------------------------
Step "Checking prerequisites"
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Fail "uv not found. Install: https://docs.astral.sh/uv/getting-started/installation/"
}
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    Fail "docker not found. Install Docker Desktop and retry."
}
docker info *> $null
if ($LASTEXITCODE -ne 0) { Fail "Docker daemon is not running." }
Write-Host "uv $(uv --version) | docker OK"

# ---- device selection --------------------------------------------------
if ($Device -eq "auto") {
    $hasGpu = [bool](Get-Command nvidia-smi -ErrorAction SilentlyContinue)
    if ($hasGpu) { nvidia-smi -L; }
    $Device = if ($hasGpu) { "gpu" } else { "cpu" }
}
Step "Installing Python dependencies (--extra $Device)"
uv sync --extra $Device
if ($LASTEXITCODE -ne 0) { Fail "uv sync failed" }

# ---- .env --------------------------------------------------------------
if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-Host ".env created from template — EDIT IT with your OpenRouter key" -ForegroundColor Yellow
}

# ---- containers --------------------------------------------------------
Step "Starting Qdrant + Postgres"
docker compose up -d
if ($LASTEXITCODE -ne 0) { Fail "docker compose up failed" }

Step "Waiting for services"
$deadline = (Get-Date).AddSeconds(90)
do {
    Start-Sleep -Seconds 2
    try { $q = (Invoke-WebRequest -Uri "http://localhost:6333/readyz" -UseBasicParsing -TimeoutSec 3).StatusCode -eq 200 }
    catch { $q = $false }
    docker exec rag-postgres pg_isready -U rag *> $null
    $p = $LASTEXITCODE -eq 0
    $qs = if ($q) { "ok" } else { "..." }
    $ps = if ($p) { "ok" } else { "..." }
    Write-Host "  qdrant: $qs  postgres: $ps"
} until (($q -and $p) -or ((Get-Date) -gt $deadline))
if (-not ($q -and $p)) { Fail "services did not become ready within 90s" }

# ---- optional ingestion ------------------------------------------------
if ($Ingest) {
    Step "Ingesting documents from data/docs (first run downloads models)"
    uv run rag ingest
}

# ---- serve ---------------------------------------------------------------
if (-not $NoServe) {
    Step "Starting API + UI at http://localhost:8000 (Ctrl+C to stop)"
    uv run rag serve
} else {
    Write-Host "`nSetup complete. Start with: uv run rag serve" -ForegroundColor Green
}
