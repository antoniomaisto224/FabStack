# FabStack Installation Guide

FabStack is a desktop application for designing semiconductor and microfabrication process flows as physically dimensioned 2D cross-sections.

## Prerequisites

- **Python**: 3.10 or newer
- **Operating System**: macOS or Windows (Linux may work but is not officially supported)
- **Disk Space**: ~500 MB for the application + dependencies

---

## Quick Installation

### Option 1: Using the Install Scripts (Recommended)

#### macOS

1. Open Terminal
2. Navigate to the project directory:
   ```bash
   cd /path/to/FabStack
   ```
3. Make the install script executable:
   ```bash
   chmod +x install_mac.sh
   ```
4. Run the installation:
   ```bash
   ./install_mac.sh
   ```
5. Start the application:
   ```bash
   ./run_mac.sh
   ```

#### Windows

1. Open PowerShell as Administrator
2. Navigate to the project directory:
   ```powershell
   cd C:\path\to\FabStack
   ```
3. Run the installation:
   ```powershell
   powershell -ExecutionPolicy Bypass -File install_windows.ps1
   ```
4. Start the application:
   ```powershell
   powershell -ExecutionPolicy Bypass -File run_windows.ps1
   ```

---

## Manual Installation

### Step 1: Create a Virtual Environment

#### macOS/Linux
```bash
python3 -m venv venv
source venv/bin/activate
```

#### Windows
```powershell
python -m venv venv
venv\Scripts\activate
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- numpy (for numerical operations)
- matplotlib (for plotting and visualization)
- PyQt5 (for the GUI)
- Pillow (for image handling)
- python-docx (for Word document export)

### Step 3: Run the Application

```bash
python app.py
```

---

## Optional: Local AI Assistant

FabStack includes an optional local AI assistant for process planning. To enable it:

### Step 1: Install the AI runtime
```bash
pip install "llama-cpp-python>=0.3,<0.4"
```

### Step 2: Download the model
Download the GGUF model file and place it in the project root directory.

**Model name**: `Mistral-7B-Instruct-v0.3-IQ4_XS.gguf`

**Download link**: https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.3-GGUF/resolve/main/Mistral-7B-Instruct-v0.3-IQ4_XS.gguf

> **Note**: The model file is ~4 GB. Without it, FabStack will use its deterministic planner.

---

## Running Tests

To verify the installation, run the test suite:

#### macOS/Linux
```bash
PYTHONDONTWRITEBYTECODE=1 ./venv/bin/python -m unittest discover -s tests -v
```

#### Windows
```powershell
$env:PYTHONDONTWRITEBYTECODE=1
.\venv\Scripts\python -m unittest discover -s tests -v
```

---

## Building Standalone Applications

### macOS Application Bundle

1. Ensure you have PyInstaller installed:
   ```bash
   pip install pyinstaller
   ```
2. Run the build script:
   ```bash
   ./build_mac_app.sh
   ```
3. The application will be created in `dist/FabStack.app` and `dist/FabStack.dmg`

### Windows Installer

1. Install [Inno Setup](https://jrsoftware.org/isdl.php)
2. Run the build script:
   ```powershell
   powershell -ExecutionPolicy Bypass -File build_windows.ps1
   ```
3. The installer will be created in `dist/FabStack-Setup.exe`

> **Note**: Windows builds must be run on a Windows machine (PyInstaller does not cross-compile).

---

## Troubleshooting

### Common Issues

#### Python version too old
Ensure you have Python 3.10 or newer:
```bash
python --version
```

#### Missing dependencies
If you get import errors, run:
```bash
pip install -r requirements.txt
```

#### PyQt5 not found
On some systems, you may need to install PyQt5 system dependencies:
- **Ubuntu/Debian**: `sudo apt-get install python3-pyqt5`
- **Fedora**: `sudo dnf install python3-qt5`

#### Application doesn't start
Check that you've activated the virtual environment:
```bash
source venv/bin/activate  # macOS/Linux
# or
venv\Scripts\activate     # Windows
```

### Getting Help

- Check the [README.md](README.md) for more information
- Review the [ARCHITECTURE.md](docs/ARCHITECTURE.md) for technical details
- Run the validation suite: `python dimension_preservation_validation.py`

---

## Directory Structure

```
FabStack/
├── fabstack/              # Main application package
│   ├── domain/           # Physical geometry and material provenance
│   ├── persistence/      # File loading and saving
│   ├── assistant/        # AI planning (deterministic and local model)
│   ├── materials/        # Chemistry labels and material library
│   ├── presentation/     # Qt window, dialogs, and UI components
│   ├── config/           # Shared visualization configuration
│   └── assets/           # Default assets and UI images
├── fabstack-web/          # Web-based viewer/server (optional)
├── tests/                # Unit, regression, and stress tests
├── docs/                 # Documentation
├── app.py                # Main entry point
├── requirements.txt      # Python dependencies
├── pyproject.toml        # Project metadata
├── install_mac.sh        # macOS installation script
├── install_windows.ps1   # Windows installation script
├── run_mac.sh            # macOS run script
└── run_windows.ps1       # Windows run script
```

---

## License

This project is provided as-is for semiconductor and microfabrication process design.

---

## Version

Current version: 1.1.0 (see [pyproject.toml](pyproject.toml))
