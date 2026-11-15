"""Audio extraction and loudness analysis."""

from __future__ import annotations

import threading
import wave
from pathlib import Path

import numpy as np

from .ffmpeg_utils import run_ffmpeg

SAMPLE_RATE = 16_000
ENERGY_HOP = 0.25  # seconds per energy frame


def extract_audio(src: Path, wav_path: Path, cancel_event: threading.Event) -> None:
    """Extract a 16 kHz mono 16-bit WAV with ffmpeg (run in the job folder)."""
    run_ffmpeg(
        ["-i", src.name, "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE), "-c:a", "pcm_s16le", wav_path.name],
        cwd=src.parent,
        cancel_event=cancel_event,
    )


def compute_energy(wav_path: Path, hop: float = ENERGY_HOP) -> np.ndarray:
    """Return RMS loudness (in dBFS, floored at -80) per ``hop`` seconds.

    The WAV is read in blocks so memory use stays flat even for multi-hour audio
    (one hour gives ~14k floats).
    """
    frames_per_hop = int(SAMPLE_RATE * hop)
    out: list[float] = []
    with wave.open(str(wav_path), "rb") as wf:
        if wf.getsampwidth() != 2:
            raise ValueError("Expected 16-bit PCM audio")
        channels = wf.getnchannels()
        while True:
            raw = wf.readframes(frames_per_hop * 240)  # ~1 minute per read
            if not raw:
                break
            samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            if channels > 1:
                samples = samples.reshape(-1, channels).mean(axis=1)
            n = len(samples) // frames_per_hop
            if n:
                block = samples[: n * frames_per_hop].reshape(n, frames_per_hop)
                rms = np.sqrt(np.mean(block ** 2, axis=1) + 1e-12)
                out.extend((20 * np.log10(rms)).tolist())
            rest = samples[n * frames_per_hop:]
            if len(rest) > frames_per_hop // 4:
                rms = float(np.sqrt(np.mean(rest ** 2) + 1e-12))
                out.append(20 * np.log10(rms))
    arr = np.asarray(out, dtype=np.float32)
    return np.maximum(arr, -80.0)
