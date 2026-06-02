# Setup script (Windows PowerShell)
# Uso: powershell -ExecutionPolicy Bypass -File scripts\setup.ps1
$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

if (-not (Test-Path ".venv")) {
    Write-Output "A criar virtualenv..."
    python -m venv .venv
}

Write-Output "A atualizar pip..."
& .\.venv\Scripts\python.exe -m pip install --upgrade pip

Write-Output "A instalar dependencias..."
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt

Write-Output "A treinar modelos..."
& .\.venv\Scripts\python.exe scripts\train_all.py

Write-Output ""
Write-Output "Pronto. Para lancar a UI:"
Write-Output "  .\.venv\Scripts\streamlit.exe run src\app\app.py"
