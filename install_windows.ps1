# Windows developer setup for FabStack.
#
# Usage: powershell -ExecutionPolicy Bypass -File install_windows.ps1

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

Write-Host "Creating virtual environment (venv)..."
python -m venv venv

$venvPython = Join-Path $PSScriptRoot "venv\Scripts\python.exe"

Write-Host "Upgrading pip..."
& $venvPython -m pip install --upgrade pip

Write-Host "Installing requirements..."
& $venvPython -m pip install -r requirements.txt

Write-Host ""
Write-Host "Installation complete."
Write-Host "To run the simulator, use: powershell -ExecutionPolicy Bypass -File run_windows.ps1"
