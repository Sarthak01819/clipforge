"""Locating and running FFmpeg, plus media probing.

FFmpeg resolution order:
1. ``ffmpeg`` on PATH
2. the static binary shipped by the ``imageio-ffmpeg`` package

Probing uses PyAV (installed with faster-whisper), so ffprobe is not required.
All subprocess calls use argument lists, never shell strings.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from .errors import JobCancelled, PipelineError

# Hide console windows spawned by ffmpeg on Windows.
_CREATE_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0


@lru_cache(maxsize=1)
def find_ffmpeg() -> str | None:
    """Return the path of a usable ffmpeg binary, or None if none is found."""
    on_path = shutil.which("ffmpeg")
    if on_path:
        return on_path
    try:
        import imageio_ffmpeg

        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe and os.path.isfile(exe):
            return exe
    except Exception:  # noqa: BLE001 - any failure means "not available"
        return None
    return None


def require_ffmpeg() -> str:
    """Return the ffmpeg path or raise a friendly error."""
    exe = find_ffmpeg()
    if not exe:
        raise PipelineError(
            "FFmpeg was not found. Install FFmpeg and add it to PATH, or run setup.bat "
            "again so the bundled copy (imageio-ffmpeg) is installed."
        )
    return exe


@dataclass
class FFmpegCapabilities:
    """What the detected ffmpeg build can do."""

    path: str | None
    version: str = ""
    has_ass: bool = False
    has_subtitles: bool = False
    has_nvenc: bool = False
    warnings: list[str] = field(default_factory=list)


def _run_quiet(args: list[str], timeout: float = 30) -> subprocess.CompletedProcess:
    return subprocess.run(
        args,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        creationflags=_CREATE_NO_WINDOW,
    )


@lru_cache(maxsize=1)
def get_capabilities() -> FFmpegCapabilities:
    """Inspect the ffmpeg build once: version, subtitle filters, NVENC."""
    exe = find_ffmpeg()
    if not exe:
        return FFmpegCapabilities(path=None, warnings=["FFmpeg not found - processing is disabled."])
    caps = FFmpegCapabilities(path=exe)
    try:
        ver = _run_quiet([exe, "-hide_banner", "-version"])
        caps.version = (ver.stdout.splitlines() or [""])[0]
        filters = _run_quiet([exe, "-hide_banner", "-filters"]).stdout
        names = {line.split()[1] for line in filters.splitlines() if len(line.split()) > 2}
        caps.has_ass = "ass" in names
        caps.has_subtitles = "subtitles" in names
    except (OSError, subprocess.SubprocessError) as exc:
        caps.warnings.append(f"Could not inspect FFmpeg: {exc}")
    if not caps.has_ass:
        caps.warnings.append(
            "This FFmpeg build has no 'ass'/'subtitles' filter (libass missing): captions "
            "cannot be burned in. Clips will still be made, and .srt files are still saved."
        )
    caps.has_nvenc = _probe_nvenc(exe)
    return caps


def _probe_nvenc(exe: str) -> bool:
    """Return True if a tiny h264_nvenc encode succeeds (NVIDIA GPU + driver present)."""
    try:
        result = _run_quiet(
            [exe, "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
             "color=c=black:s=256x256:d=0.2", "-c:v", "h264_nvenc", "-f", "null", "-"],
            timeout=20,
        )
        return result.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def run_ffmpeg(
    args: list[str],
    cwd: str | os.PathLike | None = None,
    cancel_event: threading.Event | None = None,
    timeout: float | None = None,
) -> str:
    """Run ffmpeg with ``args`` (without the binary), honouring cancellation.

    Returns stderr text on success; raises PipelineError on failure and
    JobCancelled if ``cancel_event`` is set while running.
    """
    exe = require_ffmpeg()
    cmd = [exe, "-hide_banner", "-nostdin", "-y", *args]
    # stderr goes to a temp file so a chatty ffmpeg can never fill a pipe and block.
    log_dir = Path(cwd) if cwd else Path.cwd()
    log_path = log_dir / f".ffmpeg_{threading.get_ident()}_{time.time_ns()}.log"
    started = time.monotonic()
    with open(log_path, "w+", encoding="utf-8", errors="replace") as log:
        proc = subprocess.Popen(
            cmd, cwd=cwd, stdout=subprocess.DEVNULL, stderr=log, stdin=subprocess.DEVNULL,
            creationflags=_CREATE_NO_WINDOW,
        )
        try:
            while proc.poll() is None:
                if cancel_event is not None and cancel_event.is_set():
                    proc.kill()
                    proc.wait()
                    raise JobCancelled()
                if timeout is not None and time.monotonic() - started > timeout:
                    proc.kill()
                    proc.wait()
                    raise PipelineError("FFmpeg took too long and was stopped.")
                time.sleep(0.15)
        finally:
            log.seek(0)
            stderr = log.read()
    try:
        log_path.unlink()
    except OSError:
        pass
    if proc.returncode != 0:
        tail = "\n".join(stderr.strip().splitlines()[-15:])
        raise PipelineError("FFmpeg failed while processing the video.", detail=tail)
    return stderr


@dataclass
class MediaInfo:
    """Basic facts about a media file."""

    duration: float
    has_video: bool
    has_audio: bool
    width: int = 0
    height: int = 0
    fps: float = 0.0


def probe(path: str | os.PathLike) -> MediaInfo:
    """Probe a media file with PyAV. Raises PipelineError if it is not a usable video."""
    try:
        import av
    except ImportError as exc:  # pragma: no cover - av ships with faster-whisper
        raise PipelineError("PyAV is not installed; run setup.bat again.") from exc
    try:
        with av.open(str(path)) as container:
            duration = float(container.duration / 1_000_000) if container.duration else 0.0
            vstreams = [s for s in container.streams.video]
            astreams = [s for s in container.streams.audio]
            width = height = 0
            fps = 0.0
            if vstreams:
                v = vstreams[0]
                width = int(v.codec_context.width or 0)
                height = int(v.codec_context.height or 0)
                fps = float(v.average_rate) if v.average_rate else 0.0
                if not duration and v.duration and v.time_base:
                    duration = float(v.duration * v.time_base)
            return MediaInfo(duration=duration, has_video=bool(vstreams), has_audio=bool(astreams),
                             width=width, height=height, fps=fps)
    except Exception as exc:  # noqa: BLE001 - av raises many error types
        raise PipelineError(
            "This file could not be read as a video. Please upload a common format "
            "such as MP4, MOV, MKV, WEBM or AVI.", detail=str(exc)
        ) from exc
