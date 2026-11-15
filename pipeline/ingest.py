"""Getting the source video into the job folder: uploads and yt-dlp downloads."""

from __future__ import annotations

import logging
import re
import shutil
import threading
import unicodedata
from pathlib import Path
from typing import Callable

from .errors import JobCancelled, PipelineError
from .ffmpeg_utils import find_ffmpeg

ALLOWED_EXTENSIONS = {
    ".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".flv", ".wmv", ".mpg", ".mpeg",
    ".ts", ".3gp", ".mts", ".m2ts", ".ogv",
}

ProgressFn = Callable[[float, str], None]
log = logging.getLogger(__name__)


def sanitize_filename(name: str, default: str = "video") -> str:
    """Return a safe file name: no path parts, no reserved characters, limited length."""
    name = unicodedata.normalize("NFKC", name or "")
    name = name.replace("\\", "/").split("/")[-1]
    name = re.sub(r'[<>:"|?*\x00-\x1f]', "", name)
    name = re.sub(r"\s+", "_", name).strip("._ ")
    stem, dot, ext = name.rpartition(".")
    if not dot:
        stem, ext = name, ""
    stem = re.sub(r"[^\w\-]", "_", stem)[:80].strip("_") or default
    ext = re.sub(r"[^\w]", "", ext)[:8].lower()
    reserved = {"con", "prn", "aux", "nul", *(f"com{i}" for i in range(1, 10)), *(f"lpt{i}" for i in range(1, 10))}
    if stem.lower() in reserved:
        stem = f"_{stem}"
    return f"{stem}.{ext}" if ext else stem


def check_upload_extension(filename: str) -> None:
    """Reject files whose extension is clearly not a video."""
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise PipelineError(
            f"Unsupported file type '{ext or 'none'}'. Please upload a video file "
            f"({', '.join(sorted(e.strip('.') for e in ALLOWED_EXTENSIONS))})."
        )


def is_probably_url(text: str) -> bool:
    return bool(re.match(r"^https?://[^\s/$.?#][^\s]*$", (text or "").strip(), re.IGNORECASE))


def _friendly_download_error(message: str) -> str:
    """Map yt-dlp error text to a message a non-technical user understands."""
    low = message.lower()
    if "unsupported url" in low:
        return "That link is not supported. Paste a direct link to a video page (YouTube, Vimeo, X, etc.)."
    if "private" in low:
        return "That video is private, so it cannot be downloaded."
    if "sign in" in low or "login" in low or "cookies" in low or "confirm your age" in low:
        return "The site requires sign-in (or age confirmation) for this video, so the download was blocked."
    if "http error 403" in low or "forbidden" in low:
        return "The site blocked the download (HTTP 403). Try again later, or update yt-dlp (setup.bat)."
    if "http error 404" in low or "not found" in low or "unavailable" in low:
        return "The video could not be found or is no longer available."
    if "geo" in low and "restrict" in low:
        return "This video is not available in your region."
    if "is live" in low or "live event" in low:
        return "Live streams cannot be clipped until they have finished."
    if "unable to download webpage" in low or "getaddrinfo" in low or "timed out" in low:
        return "Could not reach the site. Check your internet connection and the link."
    return "The download failed. Check the link, or update yt-dlp by running setup.bat again."


class _YtdlpLogger:
    """Send yt-dlp warnings/errors to the app log (they explain most download failures)."""

    def debug(self, msg: str) -> None:
        pass

    def info(self, msg: str) -> None:
        pass

    def warning(self, msg: str) -> None:
        log.warning("yt-dlp: %s", msg)

    def error(self, msg: str) -> None:
        log.error("yt-dlp: %s", msg)


def download_url(
    url: str,
    dest_dir: Path,
    progress: ProgressFn,
    cancel_event: threading.Event,
) -> tuple[Path, str]:
    """Download ``url`` with yt-dlp into ``dest_dir``. Returns (file path, title)."""
    try:
        import yt_dlp
    except ImportError as exc:  # pragma: no cover
        raise PipelineError("yt-dlp is not installed; run setup.bat again.") from exc

    if not is_probably_url(url):
        raise PipelineError("That does not look like a valid link. It should start with http:// or https://.")

    def hook(d: dict) -> None:
        if cancel_event.is_set():
            raise JobCancelled()
        if d.get("status") == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
            done = d.get("downloaded_bytes") or 0
            pct = (done / total * 100) if total else 0.0
            speed = d.get("speed") or 0
            msg = f"{done / 1e6:.1f} MB" + (f" of {total / 1e6:.1f} MB" if total else "")
            if speed:
                msg += f" at {speed / 1e6:.1f} MB/s"
            progress(pct, msg)
        elif d.get("status") == "finished":
            progress(100.0, "Merging audio and video")

    opts: dict = {
        "format": "bv*[height<=1080][ext=mp4]+ba[ext=m4a]/bv*[height<=1080]+ba/b[height<=1080]/bv*+ba/b",
        "merge_output_format": "mp4",
        "outtmpl": str(dest_dir / "source.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "progress_hooks": [hook],
        "restrictfilenames": True,
        "retries": 3,
        "socket_timeout": 30,
    }
    ffmpeg = find_ffmpeg()
    if ffmpeg:
        opts["ffmpeg_location"] = ffmpeg
    # YouTube needs a JS runtime to solve its download challenges; without one,
    # downloads fail with HTTP 403. Point yt-dlp at deno explicitly (the desktop
    # app bundles deno.exe and puts it on PATH).
    deno = shutil.which("deno")
    if deno:
        opts["js_runtimes"] = {"deno": {"path": deno}}
    opts["logger"] = _YtdlpLogger()

    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=True)
    except JobCancelled:
        raise
    except Exception as exc:  # noqa: BLE001 - yt-dlp raises DownloadError and friends
        if isinstance(exc.__context__, JobCancelled) or "JobCancelled" in repr(exc):
            raise JobCancelled() from exc
        raise PipelineError(_friendly_download_error(str(exc)), detail=str(exc)) from exc

    if info is None:
        raise PipelineError("The download failed: no video information was returned.")
    if info.get("_type") == "playlist":
        raise PipelineError("That link is a playlist. Paste the link of a single video.")

    candidates = sorted(
        (p for p in dest_dir.glob("source.*") if p.suffix.lower() not in {".part", ".ytdl"}),
        key=lambda p: p.stat().st_size, reverse=True,
    )
    if not candidates:
        raise PipelineError("The download finished but no video file was saved.")
    title = str(info.get("title") or "video")
    return candidates[0], title
