# -*- mode: python ; coding: utf-8 -*-

"""
PyInstaller build specification for FabStack (macOS).

Usage (from project root, inside a virtualenv with pyinstaller installed):
    pyinstaller FabStack.spec
"""

import datetime
import os
from PyInstaller.utils.hooks import collect_dynamic_libs, collect_submodules

project_root = os.path.abspath(SPECPATH)
version_namespace = {}
version_path = os.path.join(project_root, "fabstack", "version.py")
with open(version_path, encoding="utf-8") as version_file:
    exec(compile(version_file.read(), version_path, "exec"), version_namespace)
__version__ = version_namespace["__version__"]

block_cipher = None

# PyInstaller's standard hooks discover the selected Qt5Agg backend and the
# PyQt modules imported by the application. Collecting every Matplotlib/PyQt
# submodule would also bundle unrelated Tk/GTK backends, Qt Designer, tests,
# and multimedia modules.
bundle_model = (
    os.environ.get("FABSTACK_BUNDLE_MODEL", "0") == "1"
    and os.path.isfile(
        os.path.join(project_root, "Mistral-7B-Instruct-v0.3-IQ4_XS.gguf")
    )
)
hiddenimports = []
binaries = []
if bundle_model:
    try:
        hiddenimports += collect_submodules("llama_cpp")
        # llama_cpp loads its inference engine (libllama/libggml/...) via
        # ctypes at runtime, which static analysis can't see -- these must
        # be collected explicitly or the frozen app has the Python wrapper
        # but no native engine to call into.
        binaries += collect_dynamic_libs("llama_cpp")
    except Exception:
        pass

datas = [
    (os.path.join(project_root, "fabstack", "assets", "materials_library.json"), "."),
    (os.path.join(project_root, "fabstack", "assets", "method_defaults.json"), "."),
    (os.path.join(project_root, "fabstack", "assets", "fabstack_logo.png"), "."),
    (os.path.join(project_root, "fabstack", "assets", "iconlogo.png"), "."),
    (os.path.join(project_root, "fabstack", "assets", "AI button.jpg"), "."),
]
model_path = os.path.join(project_root, "Mistral-7B-Instruct-v0.3-IQ4_XS.gguf")
if bundle_model:
    datas.append((model_path, "."))

icon_path = os.path.join(project_root, "fabstack", "assets", "iconlogo.png")

a = Analysis(
    [os.path.join(project_root, "app.py")],
    pathex=[project_root],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="FabStack",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="FabStack",
)

app = BUNDLE(
    coll,
    name="FabStack.app",
    icon=icon_path,
    bundle_identifier="com.fabstack.app",
    info_plist={
        "CFBundleShortVersionString": __version__,
        "CFBundleVersion": __version__,
        "NSHighResolutionCapable": True,
        "NSHumanReadableCopyright": (
            f"Copyright © {datetime.datetime.now().year} FabStack"
        ),
    },
)
