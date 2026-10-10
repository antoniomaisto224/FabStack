# FabStack

FabStack is a desktop application for designing semiconductor and
microfabrication process flows as physically dimensioned 2D cross-sections.
The process engine retains exact dimensions in micrometers while the renderer
uses a shared adaptive scale so nanometer films and wafer-scale structures can
remain visible in the same view.

## Capabilities

- Deposit, spin-coat, bond, etch, strip, lift-off, bake, lithography, and flip
- Anisotropic and isotropic etch geometry
- Multi-wafer structural provenance and planarizer-aware rendering
- Process history, undo/redo, multi-project tabs, autosave, and `.fab` files
- Image, process-flow, and Word exports
- Deterministic process-command planning with optional local GGUF inference

## Architecture

Production code lives under the `fabstack` package and is grouped by
responsibility:

- `fabstack/domain/` owns physical geometry and material provenance.
- `fabstack/persistence/` provides validated loading and atomic saving.
- `fabstack/assistant/` contains deterministic and optional local-model planning.
- `fabstack/materials/` owns chemistry labels and material-library validation.
- `fabstack/presentation/` contains the Qt window, dialogs, and behavior mixins.
- `fabstack/config/` is the source of shared visualization limits.
- `fabstack/assets/` contains installed defaults and UI images.

The small modules at the repository root are compatibility facades for older
imports and existing developer commands. New code should import from
`fabstack.*`. See [the architecture guide](docs/ARCHITECTURE.md) for dependency
rules and a package map.

Display scaling never changes the physical process state. Main-canvas views,
exports, picking, and history rendering all consume the same display mapping.

## Development setup

FabStack requires Python 3.10 or newer. It runs on macOS and Windows.

macOS:

```bash
./install_mac.sh
./run_mac.sh
```

Windows (PowerShell):

```powershell
powershell -ExecutionPolicy Bypass -File install_windows.ps1
powershell -ExecutionPolicy Bypass -File run_windows.ps1
```

Equivalent manual setup:

```bash
python3 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
python -m pip install -r requirements.txt
python app.py
```

The optional local assistant expects
`Mistral-7B-Instruct-v0.3-IQ4_XS.gguf` in the project/application bundle. If
the model or `llama-cpp-python` is unavailable, FabStack uses its deterministic
planner. Install the optional runtime for local development with:

```bash
python -m pip install "llama-cpp-python>=0.3,<0.4"
```

## Verification

Run focused product invariants:

```bash
PYTHONDONTWRITEBYTECODE=1 ./venv/bin/python -m unittest discover -s tests -v
```

Run the full physical/rendering validation suite:

```bash
PYTHONDONTWRITEBYTECODE=1 \
QT_QPA_PLATFORM=offscreen \
MPLCONFIGDIR=/private/tmp/fabstack-mpl \
./venv/bin/python dimension_preservation_validation.py
```

Validation artifacts are written to a temporary directory and do not modify
the repository. Run the deterministic planner/UI suite with:

```bash
PYTHONDONTWRITEBYTECODE=1 \
QT_QPA_PLATFORM=offscreen \
./venv/bin/python llm_planner_stress_test.py --mock-qwen --ui-validate
```

## User data

Editable materials and application-wide presets are stored outside the app
bundle:

- macOS: `~/Library/Application Support/FabStack`
- Linux: `${XDG_DATA_HOME:-~/.local/share}/FabStack`
- Windows: `%APPDATA%\FabStack`

Set `FABSTACK_DATA_DIR` to override that location for tests or managed
deployments. Bundled defaults are copied on first use.

## Release builds

FabStack is distributed as a self-contained desktop application on both
platforms, not as a pip package or console script. The Python project
metadata is used to declare and verify dependencies for development and
release builds. The local GGUF model is excluded by default on both
platforms (the app falls back to its deterministic planner); opt in with
`FABSTACK_BUNDLE_MODEL=1` if you want it bundled.

### macOS

```bash
./build_mac_app.sh
```

The script creates `dist/FabStack.app` and `dist/FabStack.dmg`, including the
application icon and all required UI/default-data assets. Public distribution
should additionally use Apple Developer ID signing and notarization.

### Windows

PyInstaller does not cross-compile, so this must run on an actual Windows
machine (or VM) with Python 3.10+ on `PATH` and
[Inno Setup](https://jrsoftware.org/isdl.php) installed:

```powershell
powershell -ExecutionPolicy Bypass -File build_windows.ps1
```

The script creates `dist/FabStack/` (the PyInstaller build) and
`dist/FabStack-Setup.exe` (an Inno Setup installer with a Start Menu shortcut,
optional desktop shortcut, and uninstaller). If Inno Setup isn't installed,
the PyInstaller build still succeeds and `dist/FabStack/FabStack.exe` can be
zipped and distributed manually.
