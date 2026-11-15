"""Optional "GPU pack": the two CUDA 12 cuBLAS DLLs Whisper needs to run on an NVIDIA GPU.

The desktop app ships without them (they are ~740 MB) and downloads them on
request from the official ``nvidia-cublas-cu12`` wheel on PyPI. The wheel's
SHA-256 from PyPI's JSON API is verified, then only ``cublas64_12.dll`` and
``cublasLt64_12.dll`` are extracted into ``<data dir>/gpu``.
"""

from __future__ import annotations

import hashlib
import logging
import shutil
import threading
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import requests

from . import transcribe

log = logging.getLogger(__name__)

WHEEL_NAME = "nvidia-cublas-cu12"
WHEEL_VERSION = "12.9.2.10"  # CUDA 12.9 cuBLAS; tested with CTranslate2 4.9 on a GTX 1650
PYPI_JSON = f"https://pypi.org/pypi/{WHEEL_NAME}/{WHEEL_VERSION}/json"
DLLS = transcribe.CUBLAS_DLLS


@dataclass
class PackState:
    """Progress of a running install (read by the status endpoint)."""

    state: str = "idle"  # idle | downloading | extracting | done | error | cancelled
    downloaded: int = 0
    total: int = 0
    message: str = ""
    cancel: threading.Event = field(default_factory=threading.Event)


class GpuPack:
    """Install, remove and report the GPU pack in ``folder``."""

    def __init__(self, folder: Path) -> None:
        self.folder = folder
        self.progress = PackState()
        self._lock = threading.Lock()
        flag = self.folder.with_name("gpu_remove.flag")
        if flag.exists():  # a removal that Windows blocked last time (DLLs were in use)
            shutil.rmtree(self.folder, ignore_errors=True)
            if not self.folder.exists():
                flag.unlink(missing_ok=True)
        if self.installed():
            transcribe.register_dll_dir(self.folder)

    def installed(self) -> bool:
        return not self.removal_pending() and all((self.folder / name).is_file() for name in DLLS)

    def size_on_disk(self) -> int:
        return sum((self.folder / n).stat().st_size for n in DLLS if (self.folder / n).is_file())

    def status(self) -> dict:
        p = self.progress
        return {
            "nvidia_gpu": transcribe.nvidia_gpu_present(),
            "installed": self.installed(),
            "active": transcribe._cuda_available(),
            "size_mb": round(self.size_on_disk() / 1e6),
            "state": p.state,
            "downloaded_mb": round(p.downloaded / 1e6),
            "total_mb": round(p.total / 1e6),
            "message": p.message,
        }

    # ------------------------------------------------------------------ install
    def start_install(self) -> bool:
        """Start the download in a background thread. Returns False if one is already running."""
        with self._lock:
            if self.progress.state in ("downloading", "extracting"):
                return False
            self.progress = PackState(state="downloading", message="Looking up the download")
        threading.Thread(target=self._install, daemon=True, name="gpu-pack").start()
        return True

    def cancel_install(self) -> None:
        self.progress.cancel.set()

    def _wheel_info(self) -> tuple[str, str, int]:
        r = requests.get(PYPI_JSON, timeout=30)
        r.raise_for_status()
        for f in r.json()["urls"]:
            if f["filename"].endswith("win_amd64.whl"):
                return f["url"], f["digests"]["sha256"], int(f["size"])
        raise RuntimeError("No Windows build of the GPU pack was found on PyPI.")

    def _install(self) -> None:
        p = self.progress
        tmp = self.folder.parent / "gpu_download.whl.part"
        try:
            url, sha256, size = self._wheel_info()
            p.total = size
            p.message = "Downloading NVIDIA cuBLAS"
            self.folder.parent.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256()
            with requests.get(url, stream=True, timeout=60) as r, open(tmp, "wb") as fh:
                r.raise_for_status()
                for chunk in r.iter_content(chunk_size=1 << 20):
                    if p.cancel.is_set():
                        raise InterruptedError()
                    fh.write(chunk)
                    digest.update(chunk)
                    p.downloaded += len(chunk)
            if digest.hexdigest() != sha256:
                raise RuntimeError("The download was corrupted (checksum mismatch). Please try again.")

            p.state, p.message = "extracting", "Unpacking"
            staging = self.folder.with_name("gpu_staging")
            shutil.rmtree(staging, ignore_errors=True)
            staging.mkdir(parents=True)
            with zipfile.ZipFile(tmp) as zf:
                for member in zf.namelist():
                    name = member.rsplit("/", 1)[-1]
                    if name in DLLS and "/bin/" in member:
                        with zf.open(member) as src, open(staging / name, "wb") as dst:
                            shutil.copyfileobj(src, dst, length=1 << 20)
            if not all((staging / n).is_file() for n in DLLS):
                raise RuntimeError("The GPU pack did not contain the expected files.")
            shutil.rmtree(self.folder, ignore_errors=True)
            staging.rename(self.folder)
            transcribe.register_dll_dir(self.folder)
            p.state = "done"
            p.message = ("GPU acceleration is on." if transcribe._cuda_available()
                         else "Installed, but the GPU could not be used - update your NVIDIA driver.")
        except InterruptedError:
            p.state, p.message = "cancelled", "Download cancelled."
        except Exception as exc:  # noqa: BLE001 - report any failure to the UI
            log.exception("GPU pack install failed")
            p.state = "error"
            p.message = str(exc) if isinstance(exc, RuntimeError) else f"Download failed ({exc.__class__.__name__}). Check your internet connection."
        finally:
            tmp.unlink(missing_ok=True)

    # ------------------------------------------------------------------ remove
    def remove(self) -> None:
        """Delete the GPU pack. DLLs already loaded stay in use until the app restarts."""
        shutil.rmtree(self.folder, ignore_errors=True)
        pending = self.folder.exists()  # Windows keeps loaded DLLs locked
        if pending:
            self.folder.with_name("gpu_remove.flag").write_text("remove gpu pack on next start", encoding="utf-8")
        transcribe.reset_gpu_state()
        self.progress = PackState(message="Removed. The space is freed next time ClipForge starts." if pending else "Removed.")

    def removal_pending(self) -> bool:
        return self.folder.with_name("gpu_remove.flag").exists()
