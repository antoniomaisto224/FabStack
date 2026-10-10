# -*- mode: python ; coding: utf-8 -*-

"""
PyInstaller build specification for FabStack (Windows).

This must be run on Windows -- PyInstaller does not cross-compile. From a
project checkout, inside a virtualenv with pyinstaller installed:

    pyinstaller FabStack_windows.spec

See build_windows.ps1 for the full release pipeline (venv setup, PyInstaller,
and Inno Setup packaging into FabStack-Setup.exe).
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

# See FabStack.spec (macOS) for why submodule collection is scoped to
# llama_cpp only: collecting every Matplotlib/PyQt submodule would also
# bundle unrelated Tk/GTK backends, Qt Designer, tests, and multimedia
# modules that FabStack never imports.
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
        # llama_cpp loads its inference engine (llama.dll/ggml.dll/...) via
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

icon_path = os.path.join(project_root, "fabstack", "assets", "iconlogo.ico")

# Windows Explorer reads publisher/copyright info for the built .exe from an
# embedded version resource, generated here from fabstack/version.py so it
# never drifts out of sync with the actual release version.
version_tuple = tuple(int(part) for part in __version__.split("."))
version_tuple = (version_tuple + (0, 0, 0, 0))[:4]
copyright_line = f"Copyright (C) {datetime.datetime.now().year} FabStack"

version_info_path = os.path.join(project_root, "build", "fabstack_version_info.txt")
os.makedirs(os.path.dirname(version_info_path), exist_ok=True)
with open(version_info_path, "w", encoding="utf-8") as version_info_file:
    version_info_file.write(f"""VSVersionInfo(
  ffi=FixedFileInfo(
    filevers={version_tuple},
    prodvers={version_tuple},
    mask=0x3f,
    flags=0x0,
    OS=0x40004,
    fileType=0x1,
    subtype=0x0,
    date=(0, 0),
  ),
  kids=[
    StringFileInfo(
      [
        StringTable(
          u'040904B0',
          [
            StringStruct(u'CompanyName', u'FabStack'),
            StringStruct(u'FileDescription', u'FabStack process cross-section visualizer'),
            StringStruct(u'FileVersion', u'{__version__}'),
            StringStruct(u'InternalName', u'FabStack'),
            StringStruct(u'LegalCopyright', u'{copyright_line}'),
            StringStruct(u'OriginalFilename', u'FabStack.exe'),
            StringStruct(u'ProductName', u'FabStack'),
            StringStruct(u'ProductVersion', u'{__version__}'),
          ],
        )
      ]
    ),
    VarFileInfo([VarStruct(u'Translation', [1033, 1200])]),
  ],
)
""")

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
    icon=icon_path,
    version=version_info_path,
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
