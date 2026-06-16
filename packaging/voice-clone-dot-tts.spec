# PyInstaller spec for packaging the desktop app without model weights.
#
# Build from the repository root after installing dependencies:
#   pyinstaller packaging/voice-clone-dot-tts.spec --clean --noconfirm

from pathlib import Path
import sysconfig

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

ROOT = Path(SPECPATH).parent
SITE_PACKAGES = Path(sysconfig.get_paths()["purelib"])

block_cipher = None

hiddenimports = collect_submodules("dots_tts")
hiddenimports += collect_submodules("pywrapfst")
hiddenimports += ["_pywrapfst", "_pynini"]
try:
    hiddenimports += collect_submodules("mlx")
except Exception:
    pass
try:
    hiddenimports += collect_submodules("dots_tts_mlx")
except Exception:
    pass
try:
    hiddenimports += collect_submodules("torchao")
except Exception:
    pass

datas = []
datas += collect_data_files("tn", include_py_files=False)
datas += collect_data_files("itn", include_py_files=False)

binaries = []
pywrapfst_ext = next(SITE_PACKAGES.glob("_pywrapfst*.so"), None)
if pywrapfst_ext is not None:
    binaries.append((str(pywrapfst_ext), "."))

mlx_libjaccl = SITE_PACKAGES / "mlx" / "lib" / "libjaccl.dylib"
if mlx_libjaccl.exists():
    binaries.append((str(mlx_libjaccl), "."))
mlx_metallib = SITE_PACKAGES / "mlx" / "lib" / "mlx.metallib"
if mlx_metallib.exists():
    datas.append((str(mlx_metallib), "mlx/lib"))

a = Analysis(
    [str(ROOT / "packaging" / "pyinstaller_entry.py")],
    pathex=[str(ROOT / "src")],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "tests",
        "data.models",
        "data.outputs",
        "data.prompts",
    ],
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
    name="Voice Clone dots.tts",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="Voice Clone dots.tts",
)

app = BUNDLE(
    coll,
    name="Voice Clone dots.tts.app",
    icon=None,
    bundle_identifier="com.voiceclone.dots-tts",
    info_plist={
        "NSHighResolutionCapable": "True",
    },
)
