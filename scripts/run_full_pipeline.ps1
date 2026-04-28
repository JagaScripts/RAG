param(
    [switch]$SkipPrepare,
    [switch]$SkipIndex,
    [switch]$SkipApi,
    [switch]$UseUv
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Convert-ToWslPath {
    param([Parameter(Mandatory = $true)][string]$WindowsPath)

    $normalized = $WindowsPath -replace "\\", "/"
    $drive = $normalized.Substring(0, 1).ToLowerInvariant()
    $rest = $normalized.Substring(2)
    return "/mnt/$drive$rest"
}

function Run-Step {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][scriptblock]$Action
    )

    Write-Host "[RUN] $Name" -ForegroundColor Cyan
    & $Action
    Write-Host "[OK ] $Name" -ForegroundColor Green
}

$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonExe = Join-Path $projectRoot ".venv\Scripts\python.exe"
$uvExe = Get-Command uv -ErrorAction SilentlyContinue

if (-not $UseUv -and -not (Test-Path $pythonExe)) {
    throw "No se encontro Python de la venv en: $pythonExe"
}

if ($UseUv -and -not $uvExe) {
    throw "No se encontro 'uv' en PATH. Instala uv o ejecuta sin -UseUv."
}

function Run-PythonScript {
    param([Parameter(Mandatory = $true)][string]$ScriptPath)

    if ($UseUv) {
        & $uvExe.Source "run" "python" $ScriptPath | Out-Host
    }
    else {
        & $pythonExe $ScriptPath | Out-Host
    }
}

function Start-ApiProcess {
    if ($UseUv) {
        $arguments = @("run", "python", "src\\main.py")
        return Start-Process -FilePath $uvExe.Source -ArgumentList $arguments -WorkingDirectory $projectRoot -PassThru
    }

    return Start-Process -FilePath $pythonExe -ArgumentList "src\\main.py" -WorkingDirectory $projectRoot -PassThru
}

$wslProjectPath = Convert-ToWslPath -WindowsPath $projectRoot

Run-Step -Name "Arrancar Qdrant en WSL" -Action {
    $composeFile = "$wslProjectPath/docker_compose.yml"
    $cmd = @(
        "set -e",
        "mkdir -p /tmp/docker-nocreds",
        "echo '{}' > /tmp/docker-nocreds/config.json",
        "DOCKER_CONFIG=/tmp/docker-nocreds docker compose --project-directory '$wslProjectPath' -f '$composeFile' up -d qdrant"
    ) -join "`n"
    wsl -e bash -lc $cmd | Out-Host
    if ($LASTEXITCODE -ne 0) {
        throw "Fallo al arrancar Qdrant en WSL (exit code $LASTEXITCODE)."
    }
}

Run-Step -Name "Esperar health de Qdrant" -Action {
    $healthy = $false
    for ($i = 1; $i -le 25; $i++) {
        try {
            $response = Invoke-RestMethod -Uri "http://localhost:6333/healthz" -TimeoutSec 3
            if ($response -match "healthz check passed") {
                $healthy = $true
                break
            }
        }
        catch {
            Start-Sleep -Seconds 1
        }
    }

    if (-not $healthy) {
        throw "Qdrant no respondio correctamente en http://localhost:6333/healthz"
    }
}

if (-not $SkipPrepare) {
    Run-Step -Name "Generar summaries y optimized_chunks" -Action {
        Run-PythonScript -ScriptPath "$projectRoot\scripts\prepare_data.py"
    }
}

if (-not $SkipIndex) {
    Run-Step -Name "Indexar qdrantclient_index" -Action {
        Run-PythonScript -ScriptPath "$projectRoot\scripts\create_qdrant_index.py"
    }

    Run-Step -Name "Indexar langchain_index" -Action {
        Run-PythonScript -ScriptPath "$projectRoot\scripts\create_langchain_index.py"
    }

    Run-Step -Name "Indexar llamaindex_index" -Action {
        Run-PythonScript -ScriptPath "$projectRoot\scripts\create_llamaindex_indes.py"
    }
}

if (-not $SkipApi) {
    Run-Step -Name "Arrancar API (background)" -Action {
        $process = Start-ApiProcess
        Write-Host "API arrancada con PID $($process.Id) en http://localhost:8000" -ForegroundColor Yellow
    }
}

Write-Host "Pipeline completado." -ForegroundColor Green