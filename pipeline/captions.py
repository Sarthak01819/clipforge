"""Caption files (.ass for burn-in, .srt as a sidecar) built from word timestamps."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

CAPTION_SIZES = {"small": 0.055, "medium": 0.07, "large": 0.09}  # fraction of the short side
WINDOWS_FONTS = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"
# (family name, file) in order of preference. Nirmala UI covers Devanagari + Latin.
FONT_CANDIDATES = [("Nirmala UI", "Nirmala.ttc"), ("Nirmala UI", "Nirmala.ttf"), ("Nirmala UI", "NirmalaB.ttf"),
                   ("Mangal", "mangal.ttf"), ("Arial", "arialbd.ttf"), ("Arial", "arial.ttf"),
                   ("Segoe UI", "segoeuib.ttf")]


def prepare_fonts(job_dir: Path) -> str:
    """Copy caption fonts into ``job_dir/fonts`` (passed to libass as a relative fontsdir).

    Returns the primary font family name. Copying avoids handing libass a
    ``C:\\...`` path, which breaks ffmpeg's filter-argument parsing.
    """
    fonts_dir = job_dir / "fonts"
    fonts_dir.mkdir(exist_ok=True)
    family = "Arial"
    chosen = False
    for name, file in FONT_CANDIDATES:
        src = WINDOWS_FONTS / file
        if src.is_file():
            dst = fonts_dir / file
            if not dst.exists():
                try:
                    shutil.copyfile(src, dst)
                except OSError:
                    continue
            if not chosen:
                family, chosen = name, True
    return family


def group_words(words: list[dict], clip_start: float, clip_end: float, max_words: int = 4) -> list[list[dict]]:
    """Group words (absolute times) into lines of 3-4 words, with times relative to the clip."""
    rel = []
    for w in words:
        text = w["word"].strip()
        if not text or w["end"] <= clip_start or w["start"] >= clip_end:
            continue
        rel.append({"text": text, "start": max(0.0, w["start"] - clip_start),
                    "end": min(clip_end - clip_start, w["end"] - clip_start)})
    lines: list[list[dict]] = []
    current: list[dict] = []
    for idx, w in enumerate(rel):
        if current:
            gap = w["start"] - current[-1]["end"]
            ends_sentence = current[-1]["text"][-1:] in ".?!।,"
            if len(current) >= max_words or gap > 0.6 or (len(current) >= 3 and ends_sentence):
                lines.append(current)
                current = []
        current.append(w)
    if current:
        lines.append(current)
    return lines


def _ass_time(t: float) -> str:
    t = max(0.0, t)
    cs = int(round(t * 100))
    h, rem = divmod(cs, 360000)
    m, rem = divmod(rem, 6000)
    s, cs = divmod(rem, 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def _srt_time(t: float) -> str:
    ms = int(round(max(0.0, t) * 1000))
    h, rem = divmod(ms, 3600000)
    m, rem = divmod(rem, 60000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _ass_escape(text: str) -> str:
    return text.replace("\\", "/").replace("{", "(").replace("}", ")").replace("\n", " ")


def _line_spans(lines: list[list[dict]]) -> list[tuple[float, float]]:
    """Display span per line: from its first word until the next line starts (max +0.5 s)."""
    spans = []
    for k, line in enumerate(lines):
        start = line[0]["start"]
        end = line[-1]["end"] + 0.5
        if k + 1 < len(lines):
            end = min(end, lines[k + 1][0]["start"])
        spans.append((start, max(end, line[-1]["end"])))
    return spans


def write_ass(path: Path, lines: list[list[dict]], width: int, height: int, size: str = "medium",
              position: str = "bottom", font: str = "Nirmala UI") -> None:
    """Write an ASS file: bold white text, black outline, active word highlighted in yellow."""
    short = min(width, height)
    font_size = max(16, int(short * CAPTION_SIZES.get(size, CAPTION_SIZES["medium"])))
    outline = max(2, font_size // 14)
    alignment = 5 if position == "middle" else 2
    margin_v = 0 if position == "middle" else int(height * (0.14 if height > width else 0.08))
    margin_h = int(width * 0.06)
    header = (
        "[Script Info]\nScriptType: v4.00+\nWrapStyle: 0\nScaledBorderAndShadow: yes\n"
        f"PlayResX: {width}\nPlayResY: {height}\nYCbCr Matrix: TV.709\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
        "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Caption,{font},{font_size},&H00FFFFFF,&H0000FFFF,&H00000000,&H80000000,"
        f"-1,0,0,0,100,100,0,0,1,{outline},{max(1, outline // 2)},{alignment},{margin_h},{margin_h},{margin_v},1\n\n"
        "[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )
    events = []
    for line, (l_start, l_end) in zip(lines, _line_spans(lines)):
        texts = [_ass_escape(w["text"]) for w in line]
        for k, w in enumerate(line):
            w_start = l_start if k == 0 else w["start"]
            w_end = line[k + 1]["start"] if k + 1 < len(line) else l_end
            if w_end <= w_start:
                continue
            parts = [f"{{\\c&H0000E5FF&}}{t}{{\\c&H00FFFFFF&}}" if i == k else t for i, t in enumerate(texts)]
            events.append(f"Dialogue: 0,{_ass_time(w_start)},{_ass_time(w_end)},Caption,,0,0,0,,{' '.join(parts)}")
    path.write_text(header + "\n".join(events) + "\n", encoding="utf-8-sig")


def write_srt(path: Path, lines: list[list[dict]]) -> None:
    blocks = []
    for k, (line, (start, end)) in enumerate(zip(lines, _line_spans(lines)), start=1):
        text = " ".join(w["text"] for w in line)
        blocks.append(f"{k}\n{_srt_time(start)} --> {_srt_time(end)}\n{text}\n")
    path.write_text("\n".join(blocks), encoding="utf-8")
