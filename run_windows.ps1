# Launch FabStack from a source checkout on Windows.
#
# Usage: powershell -ExecutionPolicy Bypass -File run_windows.ps1

Set-Location -Path $PSScriptRoot
& (Join-Path $PSScriptRoot "venv\Scripts\python.exe") app.py
