"""Cutting and encoding clips with ffmpeg (H.264 + AAC, MP4, faststart).

Windows gotcha: the ``ass``/``subtitles`` filters cannot parse ``C:\\...`` paths
reliably, so ffmpeg always runs with ``cwd`` = the job folder and every path
passed to it (source, subtitle file, fonts dir, output) is a plain relative name.
"""

from __future__ import annotations

import threading
from pathlib import Path

from .errors import PipelineError
from .ffmpeg_utils import get_capabilities, run_ffmpeg

VERTICAL_SIZE = (1080, 1920)


def output_size(fmt: str, src_w: int, src_h: int) -> tuple[int, int]:
    """Resolution of a rendered clip for the given format."""
    if fmt == "vertical":
        return VERTICAL_SIZE
    w, h = src_w or 1280, src_h or 720
    return w - w % 2, h - h % 2


def _video_filter(fmt: str, ass_name: str | None) -> str:
    if fmt == "vertical":
        w, h = VERTICAL_SIZE
        chain = [f"scale={w}:{h}:force_original_aspect_ratio=increase", f"crop={w}:{h}", "setsar=1"]
    else:
        chain = ["scale=trunc(iw/2)*2:trunc(ih/2)*2", "setsar=1"]
    if ass_name:
        chain.append(f"ass={ass_name}:fontsdir=fonts")
    chain.append("format=yuv420p")
    return ",".join(chain)


def _encoder_args(use_nvenc: bool) -> list[str]:
    if use_nvenc:
        return ["-c:v", "h264_nvenc", "-preset", "p5", "-rc", "vbr", "-cq", "23", "-b:v", "0"]
    return ["-c:v", "libx264", "-preset", "veryfast", "-crf", "21"]


def render_clip(
    job_dir: Path,
    source_name: str,
    start: float,
    end: float,
    out_name: str,
    fmt: str,
    has_audio: bool,
    ass_name: str | None,
    cancel_event: threading.Event,
) -> None:
    """Render one clip into ``job_dir/out_name``.

    ``-ss`` before ``-i`` seeks fast; because we re-encode the cut is still
    frame accurate and output timestamps start at 0, matching the caption file.
    Falls back from NVENC to libx264 if the GPU encoder fails.
    """
    caps = get_capabilities()
    if ass_name and not caps.has_ass:
        ass_name = None
    duration = max(0.1, end - start)
    base = ["-ss", f"{start:.3f}", "-i", source_name, "-t", f"{duration:.3f}",
            "-map", "0:v:0", "-vf", _video_filter(fmt, ass_name)]
    if has_audio:
        base += ["-map", "0:a:0", "-c:a", "aac", "-b:a", "160k", "-ac", "2"]
    else:
        base += ["-an"]
    tail = ["-movflags", "+faststart", "-max_muxing_queue_size", "4096", out_name]

    encoders = [True, False] if caps.has_nvenc else [False]
    last: PipelineError | None = None
    for nvenc in encoders:
        try:
            run_ffmpeg([*base, *_encoder_args(nvenc), *tail], cwd=job_dir, cancel_event=cancel_event)
            return
        except PipelineError as exc:
            last = exc
    raise PipelineError(f"Rendering failed for {out_name}.", detail=last.detail if last else None)
