# run_tunnel.ps1
# Start FastAPI server + Cloudflare Quick Tunnel in one command.
# Auto-updates GitHub Gist so iPhone always knows the current URL.
#
# Usage:
#   .\run_tunnel.ps1           - full mode (server + tunnel + Gist)
#   .\run_tunnel.ps1 -Local    - local only (no tunnel)
#
# .env must have: GITHUB_TOKEN, GITHUB_GIST_ID, GITHUB_USERNAME

param(
    [switch]$Local
)

$ErrorActionPreference = "Stop"
$ProjectRoot = $PSScriptRoot

# ---------------------------------------------------------------
# Helper: write colored status line
# ---------------------------------------------------------------
function Write-Step { param([string]$msg) Write-Host $msg -ForegroundColor Cyan }
function Write-OK   { param([string]$msg) Write-Host "      [OK] $msg" -ForegroundColor Green }
function Write-Warn { param([string]$msg) Write-Host "      [!!] $msg" -ForegroundColor Yellow }

# ---------------------------------------------------------------
# 1. Load .env
# ---------------------------------------------------------------
$envPath = Join-Path $ProjectRoot ".env"
if (-not (Test-Path $envPath)) {
    Write-Error ".env not found: $envPath"
    exit 1
}

$env_data = @{}
foreach ($line in (Get-Content $envPath -Encoding UTF8)) {
    if ($line -match '^\s*([^#=\s][^=]*?)\s*=\s*(.+?)\s*$') {
        $env_data[$Matches[1]] = $Matches[2].Trim('"').Trim("'")
    }
}

$GH_TOKEN    = $env_data["GITHUB_TOKEN"]
$GH_GIST_ID  = $env_data["GITHUB_GIST_ID"]
$GH_USERNAME = $env_data["GITHUB_USERNAME"]

$LocalMode = $Local.IsPresent

if (-not $LocalMode) {
    $missing = @()
    if (-not $GH_TOKEN)    { $missing += "GITHUB_TOKEN" }
    if (-not $GH_GIST_ID)  { $missing += "GITHUB_GIST_ID" }
    if (-not $GH_USERNAME) { $missing += "GITHUB_USERNAME" }
    if ($missing.Count -gt 0) {
        Write-Warn "Missing in .env: $($missing -join ', ') -> switching to local mode."
        $LocalMode = $true
    }
}

# ---------------------------------------------------------------
# 2. Check / Download cloudflared.exe
# ---------------------------------------------------------------
$cfExe = $null
if (-not $LocalMode) {
    $cfInPath = Get-Command "cloudflared" -ErrorAction SilentlyContinue
    $cfLocal  = Join-Path $ProjectRoot "cloudflared.exe"

    if ($cfInPath) {
        $cfExe = "cloudflared"
        Write-OK "cloudflared found in PATH."
    } elseif (Test-Path $cfLocal) {
        $cfExe = $cfLocal
        Write-OK "cloudflared.exe found in project root."
    } else {
        Write-Host "[DL] Downloading cloudflared.exe..." -ForegroundColor Yellow
        $dlUrl = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe"
        try {
            Invoke-WebRequest -Uri $dlUrl -OutFile $cfLocal -UseBasicParsing
            $cfExe = $cfLocal
            Write-OK "cloudflared.exe downloaded to project root."
        } catch {
            Write-Warn "Download failed: $_ -> switching to local mode."
            $LocalMode = $true
        }
    }
}

# ---------------------------------------------------------------
# 3. Start FastAPI server (background process)
# ---------------------------------------------------------------
$pythonExe = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$runPyPath = Join-Path $ProjectRoot "run.py"

if (-not (Test-Path $pythonExe)) {
    Write-Error "Python venv not found: $pythonExe"
    exit 1
}

Write-Step ""
Write-Step "[1/3] Starting FastAPI server..."

$serverProc = Start-Process `
    -FilePath   $pythonExe `
    -ArgumentList ("`"" + $runPyPath + "`"") `
    -WorkingDirectory $ProjectRoot `
    -NoNewWindow `
    -PassThru

Write-Host "      PID: $($serverProc.Id) | Waiting" -NoNewline

$ready  = $false
$waited = 0
while ($waited -lt 30) {
    Start-Sleep 1
    $waited++
    Write-Host "." -NoNewline
    try {
        $null = Invoke-WebRequest "http://localhost:8000/" `
            -TimeoutSec 1 -UseBasicParsing -ErrorAction Stop
        $ready = $true
        break
    } catch { }
}
Write-Host ""

if ($ready) {
    Write-OK "Server ready at http://localhost:8000"
} else {
    Write-Warn "Server not responding yet (still loading). Continuing..."
}

# ---------------------------------------------------------------
# 4. Start Cloudflare Quick Tunnel
# ---------------------------------------------------------------
$tunnelUrl = $null
$cfProc    = $null

if (-not $LocalMode) {
    Write-Step "[2/3] Starting Cloudflare Quick Tunnel..."

    $cfLogFile = Join-Path $env:TEMP ("cf_manga_" + $PID + ".log")
    if (Test-Path $cfLogFile) { Remove-Item $cfLogFile -Force }

    $cfProc = Start-Process `
        -FilePath   $cfExe `
        -ArgumentList "tunnel --url http://localhost:8000" `
        -NoNewWindow `
        -PassThru `
        -RedirectStandardError $cfLogFile

    Write-Host "      PID: $($cfProc.Id) | Waiting for URL" -NoNewline

    $deadline = (Get-Date).AddSeconds(45)
    while ((Get-Date) -lt $deadline) {
        Start-Sleep 1
        Write-Host "." -NoNewline
        if (Test-Path $cfLogFile) {
            $logText = Get-Content $cfLogFile -Raw -ErrorAction SilentlyContinue
            if ($logText -match "https://[a-z0-9\-]+\.trycloudflare\.com") {
                $tunnelUrl = $Matches[0]
                break
            }
        }
    }
    Write-Host ""

    if ($tunnelUrl) {
        Write-OK "Tunnel URL: $tunnelUrl"
    } else {
        Write-Warn "Could not get tunnel URL after 45s."
    }
}

# ---------------------------------------------------------------
# 5. Update GitHub Gist
# ---------------------------------------------------------------
if ($tunnelUrl) {
    Write-Step "[3/3] Updating GitHub Gist..."

    $gistBody = '{"files":{"backend_url.txt":{"content":"' + $tunnelUrl + '"}}}'
    $gistHeaders = @{
        "Authorization" = "token " + $GH_TOKEN
        "Accept"        = "application/vnd.github.v3+json"
        "Content-Type"  = "application/json"
    }

    try {
        $null = Invoke-RestMethod `
            -Uri     ("https://api.github.com/gists/" + $GH_GIST_ID) `
            -Method  Patch `
            -Body    $gistBody `
            -Headers $gistHeaders
        Write-OK "Gist updated. iPhone will auto-detect new URL."
    } catch {
        Write-Warn "Gist update failed: $_ -> iPhone needs manual URL: $tunnelUrl"
    }
} elseif (-not $LocalMode) {
    Write-Step "[3/3] Skipped (no tunnel URL available)."
}

# ---------------------------------------------------------------
# 6. Status banner
# ---------------------------------------------------------------
Write-Host ""
Write-Host "=====================================================" -ForegroundColor Cyan
Write-Host "  READY!  Press Ctrl+C to stop everything." -ForegroundColor Green
Write-Host ""
if ($tunnelUrl) {
    Write-Host "  iPhone URL : $tunnelUrl" -ForegroundColor Yellow
    $rawUrl = "https://gist.githubusercontent.com/" + $GH_USERNAME + "/" + $GH_GIST_ID + "/raw/backend_url.txt"
    Write-Host "  Gist URL   : $rawUrl"
}
Write-Host "  Local URL  : http://localhost:8000" -ForegroundColor White
Write-Host "=====================================================" -ForegroundColor Cyan
Write-Host ""

# ---------------------------------------------------------------
# 7. Keep alive + auto-restart server if it crashes
# ---------------------------------------------------------------
try {
    while ($true) {
        Start-Sleep 2
        if ($serverProc -and $serverProc.HasExited) {
            Write-Warn "Server exited (code=$($serverProc.ExitCode)). Restarting..."
            $serverProc = Start-Process `
                -FilePath $pythonExe -ArgumentList ("`"" + $runPyPath + "`"") `
                -WorkingDirectory $ProjectRoot -NoNewWindow -PassThru
            Write-Host "      New PID: $($serverProc.Id)" -ForegroundColor Yellow
        }
    }
} finally {
    Write-Host ""
    Write-Host "Shutting down..." -ForegroundColor Yellow
    if ($serverProc -and -not $serverProc.HasExited) {
        Stop-Process -Id $serverProc.Id -Force -ErrorAction SilentlyContinue
    }
    if ($cfProc -and -not $cfProc.HasExited) {
        Stop-Process -Id $cfProc.Id -Force -ErrorAction SilentlyContinue
    }
    Write-Host "Done." -ForegroundColor Green
}
