# PyInstaller recipe for the ClipForge desktop app.  Build with build.bat.
# Produces dist\ClipForge\ClipForge.exe (a folder build: starts faster than a
# single-file exe, and the installer compresses it anyway).
# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules, copy_metadata

ROOT = Path(SPECPATH)
VENV_SCRIPTS = ROOT / ".venv" / "Scripts"

datas = [(str(ROOT / "static"), "static")]
datas += collect_data_files("faster_whisper")           # Silero VAD model (assets/)
datas += collect_data_files("imageio_ffmpeg", include_py_files=False)  # bundled ffmpeg.exe
datas += collect_data_files("yt_dlp_ejs")               # YouTube JS challenge solver scripts (imported dynamically,
datas += copy_metadata("yt_dlp_ejs")                    # so its modules are listed in hiddenimports too)

binaries = collect_dynamic_libs("ctranslate2")          # ctranslate2.dll, OpenMP runtime
deno = VENV_SCRIPTS / "deno.exe"                        # JS runtime yt-dlp needs for YouTube
if deno.is_file():
    binaries.append((str(deno), "."))

hiddenimports = collect_submodules("uvicorn") + collect_submodules("yt_dlp") + collect_submodules("yt_dlp_ejs") + ["app"]

excludes = [
    "nvidia",            # 1.95 GB of CUDA libraries: offered as an optional in-app download instead
    "hf_xet",            # optional faster HF downloads; plain HTTPS works fine
    "tkinter", "unittest", "pydoc", "doctest", "pip", "setuptools", "wheel",
    "IPython", "matplotlib", "torch", "tensorflow", "PIL", "pytest",
]

a = Analysis(
    [str(ROOT / "desktop.py")],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=excludes,
    noarchive=False,
    optimize=1,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="ClipForge",
    icon=str(ROOT / "assets" / "clipforge.ico"),
    console=False,       # windowed app: no terminal window
    upx=False,           # UPX-packed exes trigger antivirus false positives
    version=str(ROOT / "assets" / "version_info.txt"),
)

coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name="ClipForge")
