"""Speech-to-text with faster-whisper (word timestamps, VAD filter).

Uses an NVIDIA GPU automatically when the CUDA 12 cuBLAS DLLs can be loaded,
otherwise CPU with int8. If the GPU path fails it silently retries on CPU.

CTranslate2 (4.5+) only needs cuBLAS (cublas64_12.dll + cublasLt64_12.dll) to
run Whisper on the GPU; cuDNN is not required. The DLLs can come from the
``nvidia-cublas-cu12`` pip wheel or from a folder registered with
``register_dll_dir`` (the desktop app's downloadable GPU pack).
"""

from __future__ import annotations

import logging
import threading
import wave
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .errors import JobCancelled, PipelineError

log = logging.getLogger(__name__)

WHISPER_MODELS = ("tiny", "base", "small", "medium")
ProgressFn = Callable[[float, str], None]

_model_cache: dict[tuple[str, str, str], Any] = {}
_model_lock = threading.Lock()


_gpu_broken = False  # set after a GPU failure so later jobs go straight to CPU
_extra_dll_dirs: list[Path] = []

CUBLAS_DLLS = ("cublas64_12.dll", "cublasLt64_12.dll")


def _add_dll_dir(bin_dir: Path) -> None:
    import os

    if bin_dir.is_dir() and str(bin_dir) not in os.environ.get("PATH", ""):
        os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")
        if hasattr(os, "add_dll_directory"):
            try:
                os.add_dll_directory(str(bin_dir))
            except OSError:
                pass


def register_dll_dir(path: Path) -> None:
    """Add a folder holding CUDA DLLs (e.g. the app's GPU pack) and re-check GPU support."""
    if path not in _extra_dll_dirs:
        _extra_dll_dirs.append(path)
    reset_gpu_state()


def reset_gpu_state() -> None:
    """Forget cached GPU checks (after installing or removing the GPU pack)."""
    global _gpu_broken
    _gpu_broken = False
    _cuda_usable.cache_clear()


def _add_nvidia_dll_dirs() -> None:
    """Make CUDA DLLs loadable: registered folders, then the optional nvidia-* pip wheels."""
    for d in _extra_dll_dirs:
        _add_dll_dir(d)
    try:
        import nvidia  # namespace package installed by the nvidia-* wheels
    except ImportError:
        return
    for base in getattr(nvidia, "__path__", []):
        for bin_dir in Path(base).glob("*/bin"):
            _add_dll_dir(bin_dir)


def nvidia_gpu_present() -> bool:
    """True if CTranslate2 sees an NVIDIA GPU (driver installed), regardless of cuBLAS."""
    try:
        import ctranslate2

        return ctranslate2.get_cuda_device_count() > 0
    except Exception:  # noqa: BLE001
        return False


def _cuda_libs_loadable() -> bool:
    """True if the CUDA 12 cuBLAS DLLs that CTranslate2 needs can be loaded.

    Checked up front because a failed GPU attempt can leave CTranslate2 in a bad
    state (we observed a hang when retrying after "cublas64_12.dll not found").
    """
    import ctypes
    import sys

    _add_nvidia_dll_dirs()
    names = CUBLAS_DLLS if sys.platform == "win32" else ("libcublas.so.12", "libcublasLt.so.12")
    for name in names:
        try:
            ctypes.CDLL(name)
        except OSError:
            return False
    return True


@lru_cache(maxsize=1)
def _cuda_usable() -> bool:
    try:
        return nvidia_gpu_present() and _cuda_libs_loadable()
    except Exception:  # noqa: BLE001
        return False


def _cuda_available() -> bool:
    """Whether Whisper will try the NVIDIA GPU."""
    return not _gpu_broken and _cuda_usable()


def _load_model(name: str, device: str, compute_type: str, models_dir: Path):
    from faster_whisper import WhisperModel

    key = (name, device, compute_type)
    with _model_lock:
        if key not in _model_cache:
            _model_cache.clear()  # keep at most one model in memory
            _model_cache[key] = WhisperModel(
                name, device=device, compute_type=compute_type, download_root=str(models_dir)
            )
        return _model_cache[key]


def _load_wav(wav_path: Path) -> np.ndarray:
    """Load our 16 kHz mono 16-bit WAV as float32.

    Passing an array (instead of a path) bypasses faster-whisper's PyAV-based
    decoder, which breaks with some PyAV versions. One hour is ~230 MB.
    """
    with wave.open(str(wav_path), "rb") as wf:
        if wf.getframerate() != 16000 or wf.getnchannels() != 1 or wf.getsampwidth() != 2:
            raise ValueError("Expected 16 kHz mono 16-bit WAV")
        n = wf.getnframes()
        out = np.empty(n, dtype=np.float32)
        pos = 0
        while pos < n:
            raw = wf.readframes(16000 * 60)
            if not raw:
                break
            block = np.frombuffer(raw, dtype=np.int16)
            out[pos: pos + len(block)] = block / 32768.0
            pos += len(block)
    return out[:pos]


def _run(
    model, wav_path: Path, language: str | None, progress: ProgressFn, cancel_event: threading.Event
) -> dict:
    segments_iter, info = model.transcribe(
        _load_wav(wav_path),
        language=language,
        word_timestamps=True,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
        beam_size=5,
        condition_on_previous_text=False,
    )
    duration = float(info.duration or 0) or 1.0
    segments: list[dict] = []
    for seg in segments_iter:  # generator: transcription happens as we iterate
        if cancel_event.is_set():
            raise JobCancelled()
        words = [
            {"start": round(float(w.start), 3), "end": round(float(w.end), 3), "word": w.word}
            for w in (seg.words or [])
        ]
        text = seg.text.strip()
        if text:
            segments.append({"start": round(float(seg.start), 3), "end": round(float(seg.end), 3),
                             "text": text, "words": words})
        progress(min(99.0, seg.end / duration * 100), text[:80])
    return {"language": info.language, "language_probability": float(info.language_probability or 0),
            "duration": duration, "segments": segments}


def transcribe(
    wav_path: Path,
    model_name: str,
    language: str | None,
    models_dir: Path,
    progress: ProgressFn,
    cancel_event: threading.Event,
) -> dict:
    """Transcribe ``wav_path``. Returns a dict with language and segments (each with words)."""
    if model_name not in WHISPER_MODELS:
        model_name = "base"
    language = (language or "").strip().lower() or None
    if language == "auto":
        language = None
    models_dir.mkdir(parents=True, exist_ok=True)

    global _gpu_broken
    attempts: list[tuple[str, str]] = []
    if _cuda_available():
        _add_nvidia_dll_dirs()
        attempts.append(("cuda", "int8_float32"))
    attempts.append(("cpu", "int8"))

    last_error: Exception | None = None
    for device, compute_type in attempts:
        try:
            progress(0.0, f"Loading Whisper '{model_name}' on {device.upper()} (first run downloads it)")
            model = _load_model(model_name, device, compute_type, models_dir)
            result = _run(model, wav_path, language, progress, cancel_event)
            result["device"] = device
            return result
        except JobCancelled:
            raise
        except Exception as exc:  # noqa: BLE001 - GPU failures come in many shapes
            log.warning("Whisper on %s failed: %s", device, exc)
            last_error = exc
            if device == "cuda":
                _gpu_broken = True
            with _model_lock:
                _model_cache.clear()
    raise PipelineError(
        "Transcription failed. If this is the first run, check your internet connection "
        "(the Whisper model must be downloaded once).", detail=str(last_error)
    )
