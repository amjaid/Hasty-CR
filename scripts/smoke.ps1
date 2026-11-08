$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$hastyPython = Join-Path $projectRoot ".venv\clashai\Scripts\python.exe"
$pyClashBotPython = Join-Path $projectRoot ".venv\pyclashbot\Scripts\python.exe"

Push-Location $projectRoot
try {
    & $hastyPython -m pytest -q
} finally {
    Pop-Location
}

Push-Location (Join-Path $projectRoot "vendor\py-clash-bot")
try {
    & $pyClashBotPython -m pytest -q
} finally {
    Pop-Location
}
