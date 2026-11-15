"""End-to-end job orchestration. Independent of any web framework.

Usage::

    opts = JobOptions(num_clips=5, captions=True, fmt="vertical")
    result = run_job(job_dir, source, opts, report=print_progress, cancel_event=threading.Event())
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

from . import ai_picker, audio, captions, ingest, render, scoring, transcribe
from .errors import JobCancelled, PipelineError
from .ffmpeg_utils import get_capabilities, probe, require_ffmpeg

log = logging.getLogger(__name__)

# report(stage, percent_of_stage, message, overall_percent)
ReportFn = Callable[[str, float, str, float], None]

# Overall progress budget per stage (start, end) in percent.
STAGES = {
    "Downloading": (0, 15),
    "Checking video": (15, 17),
    "Extracting audio": (17, 22),
    "Transcribing": (22, 70),
    "Finding highlights": (70, 75),
    "Rendering": (75, 99),
}


@dataclass
class JobOptions:
    num_clips: int = 5
    min_len: float = 20.0
    max_len: float = 60.0
    fmt: str = "original"  # "original" | "vertical"
    captions: bool = True
    caption_size: str = "medium"  # small | medium | large
    caption_position: str = "bottom"  # bottom | middle
    whisper_model: str = "base"
    language: str = "auto"
    use_ai: bool = False
    ollama: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> "JobOptions":
        o = cls()
        o.num_clips = int(min(20, max(1, int(d.get("num_clips", o.num_clips)))))
        o.min_len = float(min(300, max(5, float(d.get("min_len", o.min_len)))))
        o.max_len = float(min(600, max(o.min_len + 5, float(d.get("max_len", o.max_len)))))
        o.fmt = "vertical" if d.get("fmt") == "vertical" else "original"
        o.captions = bool(d.get("captions", o.captions))
        o.caption_size = d.get("caption_size") if d.get("caption_size") in captions.CAPTION_SIZES else "medium"
        o.caption_position = "middle" if d.get("caption_position") == "middle" else "bottom"
        o.whisper_model = d.get("whisper_model") if d.get("whisper_model") in transcribe.WHISPER_MODELS else "base"
        lang = str(d.get("language") or "auto").strip().lower()
        o.language = lang if lang.isalpha() and len(lang) <= 8 else "auto"
        o.use_ai = bool(d.get("use_ai", False))
        o.ollama = dict(d.get("ollama") or {})
        return o


@dataclass
class Source:
    """Either an already-saved file in the job folder or a URL to download."""

    file: Path | None = None
    url: str | None = None
    original_name: str = ""


class _Reporter:
    def __init__(self, report: ReportFn) -> None:
        self.report = report

    def __call__(self, stage: str, pct: float, msg: str = "") -> None:
        lo, hi = STAGES.get(stage.split(" clip")[0], (0, 100))
        pct = max(0.0, min(100.0, pct))
        self.report(stage, pct, msg, lo + (hi - lo) * pct / 100)

    def stage(self, stage: str) -> Callable[[float, str], None]:
        return lambda pct, msg="": self(stage, pct, msg)


def _fmt_time(t: float) -> str:
    m, s = divmod(int(round(t)), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def run_job(
    job_dir: Path,
    source: Source,
    opts: JobOptions,
    report: ReportFn,
    cancel_event: threading.Event,
    models_dir: Path,
) -> dict:
    """Run the full pipeline. Returns the result dict (also saved as result.json)."""
    rep = _Reporter(report)
    require_ffmpeg()
    caps = get_capabilities()
    notices: list[str] = []
    t0 = time.time()

    def check_cancel() -> None:
        if cancel_event.is_set():
            raise JobCancelled()

    # 1. Ingest -----------------------------------------------------------------
    title = source.original_name or "video"
    if source.url:
        rep("Downloading", 0, "Contacting site")
        src, title = ingest.download_url(source.url, job_dir, rep.stage("Downloading"), cancel_event)
    elif source.file and source.file.is_file():
        src = source.file
    else:
        raise PipelineError("No video was provided.")
    check_cancel()

    rep("Checking video", 0, "Reading file")
    info = probe(src)
    if not info.has_video:
        raise PipelineError("The file has no video stream. Please provide a video file.")
    if info.duration <= 1:
        raise PipelineError("The video is too short to clip.")
    rep("Checking video", 100, f"{info.width}x{info.height}, {_fmt_time(info.duration)}")

    # 2. Audio ------------------------------------------------------------------
    transcript: dict = {"language": None, "segments": []}
    energy = None
    units: list[scoring.Unit] = []
    if info.has_audio:
        rep("Extracting audio", 0, "16 kHz mono WAV")
        wav = job_dir / "audio.wav"
        audio.extract_audio(src, wav, cancel_event)
        energy = audio.compute_energy(wav)
        rep("Extracting audio", 100, "Done")
        check_cancel()

        # 3. Transcribe ---------------------------------------------------------
        transcript = transcribe.transcribe(wav, opts.whisper_model, opts.language, models_dir,
                                           rep.stage("Transcribing"), cancel_event)
        (job_dir / "transcript.json").write_text(json.dumps(transcript, ensure_ascii=False, indent=1), encoding="utf-8")
        rep("Transcribing", 100, f"Language: {transcript.get('language')} ({transcript.get('device', 'cpu').upper()})")
        units = scoring.build_units(transcript["segments"])
        try:
            wav.unlink()  # can be large for long videos; energy is already computed
        except OSError:
            pass
    else:
        notices.append("The video has no audio track, so clips were picked evenly and captions were skipped.")
    check_cancel()

    # 4-8. Find highlights ------------------------------------------------------
    rep("Finding highlights", 0, "Scoring candidates")
    import numpy as np

    energy_arr = energy if energy is not None else np.zeros(0, dtype=np.float32)
    n_words = sum(len(u.words) for u in units)
    speech_time = sum(u.end - u.start for u in units)
    low_speech = n_words < 25 or speech_time < max(10.0, 0.1 * info.duration)
    scorer = scoring.Scorer(energy_arr)
    picker = "heuristic"
    if low_speech:
        if info.has_audio:
            notices.append("Little or no speech was detected, so clips were picked by audio energy and captions were skipped.")
        clips = scoring.energy_only_clips(energy_arr, opts.num_clips, opts.min_len, opts.max_len, info.duration)
        picker = "energy"
    else:
        clips = scoring.heuristic_clips(units, scorer, opts.num_clips, opts.min_len, opts.max_len, info.duration)
        if opts.use_ai:
            settings = ai_picker.OllamaSettings.from_dict(opts.ollama)
            try:
                clips = ai_picker.pick_clips(units, scorer, settings, opts.num_clips, opts.min_len, opts.max_len,
                                             info.duration, cancel_event, rep.stage("Finding highlights"))
                picker = "ai"
            except ai_picker.AIPickError as exc:
                notices.append(f"AI clip-picker unavailable ({exc}) - used the built-in heuristic instead.")
            except JobCancelled:
                raise
            except Exception as exc:  # noqa: BLE001 - the AI step must never crash the job
                log.exception("AI picker crashed")
                notices.append(f"AI clip-picker failed ({exc.__class__.__name__}) - used the built-in heuristic instead.")
    if not clips:
        raise PipelineError("No clips could be found in this video. Try a shorter minimum length.")
    rep("Finding highlights", 100, f"{len(clips)} clip(s) selected")
    check_cancel()

    # 9. Render -------------------------------------------------------------------
    want_captions = opts.captions and not low_speech and bool(units)
    if opts.captions and want_captions and not caps.has_ass:
        notices.append("Captions could not be burned in (this FFmpeg has no libass); .srt files were still saved.")
    font = captions.prepare_fonts(job_dir) if want_captions and caps.has_ass else "Arial"
    out_w, out_h = render.output_size(opts.fmt, info.width, info.height)
    results: list[dict] = []
    for k, clip in enumerate(clips, start=1):
        check_cancel()
        stage = f"Rendering clip {k} of {len(clips)}"
        rep(stage, (k - 1) / len(clips) * 100, clip.title)
        base = f"clip_{k:02d}"
        ass_name = None
        srt_name = None
        if want_captions:
            lines = captions.group_words(clip.words, clip.start, clip.end)
            if lines:
                srt_name = f"{base}.srt"
                captions.write_srt(job_dir / srt_name, lines)
                if caps.has_ass:
                    ass_name = f"{base}.ass"
                    captions.write_ass(job_dir / ass_name, lines, out_w, out_h, opts.caption_size,
                                       opts.caption_position, font)
        render.render_clip(job_dir, src.name, clip.start, clip.end, f"{base}.mp4", opts.fmt,
                           info.has_audio, ass_name, cancel_event)
        results.append({
            "index": k,
            "file": f"{base}.mp4",
            "srt": srt_name,
            "title": clip.title,
            "reason": clip.reason,
            "start": clip.start,
            "end": clip.end,
            "start_label": _fmt_time(clip.start),
            "end_label": _fmt_time(clip.end),
            "duration": round(clip.end - clip.start, 2),
            "score": clip.score,
            "text": clip.text,
            "source": clip.source,
            "features": clip.features,
        })
        rep(stage, k / len(clips) * 100, "Done")

    result = {
        "title": title,
        "source_file": src.name,
        "duration": info.duration,
        "width": info.width,
        "height": info.height,
        "output_size": [out_w, out_h],
        "language": transcript.get("language"),
        "device": transcript.get("device"),
        "picker": picker,
        "notices": notices,
        "clips": results,
        "options": {k: v for k, v in asdict(opts).items() if k != "ollama"},
        "elapsed_sec": round(time.time() - t0, 1),
    }
    (job_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    return result
