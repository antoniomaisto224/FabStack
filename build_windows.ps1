# Windows release build for FabStack.
#
# Must be run on Windows -- PyInstaller does not cross-compile from macOS.
# Requires Python 3.10+ on PATH and Inno Setup (https://jrsoftware.org/isdl.php)
# installed for the final Setup.exe packaging step.
#
# Usage (from an elevated or normal PowerShell prompt):
#   powershell -ExecutionPolicy Bypass -File build_windows.ps1
#
# To bundle the optional multi-gigabyte local GGUF model, set
# FABSTACK_BUNDLE_MODEL=1 in the environment before running this script.

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

$AppName = "FabStack"

Write-Host ">>> Creating / updating virtual environment..."
python -m venv venv

$venvPython = Join-Path $PSScriptRoot "venv\Scripts\python.exe"

Write-Host ">>> Upgrading pip..."
& $venvPython -m pip install --upgrade pip

Write-Host ">>> Installing Python dependencies (including PyInstaller)..."
& $venvPython -m pip install -r requirements.txt "pyinstaller>=6.10,<7"

if ($env:FABSTACK_BUNDLE_MODEL -eq "1") {
    Write-Host ">>> Installing optional local-AI runtime..."
    & $venvPython -m pip install "llama-cpp-python>=0.3,<0.4"
}

Write-Host ">>> Building Windows executable with PyInstaller..."
& $venvPython -m PyInstaller --clean --noconfirm FabStack_windows.spec

$distApp = Join-Path $PSScriptRoot "dist\$AppName\$AppName.exe"
if (-not (Test-Path $distApp)) {
    Write-Error "ERROR: $distApp not found after build."
    exit 1
}

Write-Host ">>> Locating Inno Setup compiler (ISCC.exe)..."
$iscc = Get-Command ISCC.exe -ErrorAction SilentlyContinue
if (-not $iscc) {
    $candidates = @(
        "$Env:ProgramFiles(x86)\Inno Setup 6\ISCC.exe",
        "$Env:ProgramFiles\Inno Setup 6\ISCC.exe"
    )
    foreach ($candidate in $candidates) {
        if (Test-Path $candidate) {
            $iscc = $candidate
            break
        }
    }
}
if (-not $iscc) {
    Write-Error @"
ERROR: Inno Setup (ISCC.exe) was not found.
Install it from https://jrsoftware.org/isdl.php, then re-run this script.
The FabStack.exe build itself succeeded -- dist\$AppName\$AppName.exe is ready
to zip and distribute manually if you'd rather skip the installer wizard.
"@
    exit 1
}

Write-Host ">>> Reading FabStack version..."
$version = & $venvPython -c "import runpy; ns = runpy.run_path('fabstack/version.py'); print(ns['__version__'])"
$year = (Get-Date).Year
Write-Host "    Version: $version"

Write-Host ">>> Building Windows installer with Inno Setup..."
& $iscc "/DMyAppVersion=$version" "/DMyAppYear=$year" installer.iss

$setupExe = Join-Path $PSScriptRoot "dist\FabStack-Setup.exe"
if (-not (Test-Path $setupExe)) {
    Write-Error "ERROR: $setupExe not found after Inno Setup build."
    exit 1
}

Write-Host ""
Write-Host "Build complete."
Write-Host "You can distribute dist\FabStack-Setup.exe to Windows users."
Write-Host "End users just need to:"
Write-Host "  1) Run FabStack-Setup.exe"
Write-Host "  2) Follow the install wizard"
Write-Host "  3) Launch FabStack from the Start Menu or desktop shortcut"
