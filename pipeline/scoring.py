"""Candidate clip generation and the built-in engagement heuristic.

Score (0-100) combines:
* loudness of the window relative to the whole video (35 %)
* speech pace in words per second (20 %)
* hook signals in the text - strong phrases, questions, exclamations, numbers (30 %)
* lack of long silences (15 %; long pauses pull the score down)
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import numpy as np

from .audio import ENERGY_HOP

PAD = 0.3  # seconds of padding before/after a clip

STRONG_PHRASES = [
    # English
    "the secret", "secret is", "nobody tells you", "no one tells you", "nobody talks about",
    "here's why", "here is why", "this is why", "that's why", "biggest mistake", "the mistake",
    "the truth", "truth is", "you won't believe", "you need to", "you have to", "the reason",
    "the problem", "most people", "never do", "stop doing", "imagine", "what if", "the key",
    "game changer", "let me tell you", "the best", "the worst", "how to", "important",
    "crazy", "insane", "shocking", "mind blowing", "the real", "listen", "trust me",
    "number one", "first thing", "life changing", "changed my life", "hack", "tip",
    # Hindi (Devanagari)
    "राज़", "राज", "रहस्य", "कोई नहीं बताता", "कोई नहीं बताएगा", "सबसे बड़ी गलती", "बड़ी गलती",
    "गलती", "सच्चाई", "सच", "यही कारण", "इसलिए", "ज़रूरी", "जरूरी", "सबसे अच्छा", "सबसे बुरा",
    "कभी नहीं", "सोचिए", "सुनिए", "ध्यान दीजिए", "असली", "कमाल", "हैरान", "चौंकाने",
    "ज़िंदगी बदल", "जिंदगी बदल", "सबसे ज़्यादा", "सबसे ज्यादा",
    # Hindi (romanised / Hinglish)
    "sabse badi galti", "koi nahi batata", "koi nahin batata", "raaz", "sach", "asli",
    "dhyan dijiye", "suniye", "sochiye", "kamaal",
]
QUESTION_WORDS = ["why", "how", "what", "when", "who", "क्या", "क्यों", "कैसे", "कब", "कौन", "kya", "kyun", "kaise"]
NUMBER_WORDS = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
                "hundred", "thousand", "million", "billion", "percent", "एक", "दो", "तीन", "चार",
                "पांच", "पाँच", "दस", "सौ", "हज़ार", "हजार", "लाख", "करोड़"]

_SENTENCE_END = re.compile(r"[.?!।॥]+[\"')\]]*$")
_NUMBER_RE = re.compile(r"[0-9०-९]+")


def _phrase_regex(phrases: list[str]) -> re.Pattern:
    # Word boundaries for Latin text; Devanagari is matched as a plain substring.
    parts = []
    for p in sorted(phrases, key=len, reverse=True):
        esc = re.escape(p)
        parts.append(rf"\b{esc}\b" if p.isascii() else esc)
    return re.compile("|".join(parts), re.IGNORECASE)


_STRONG_RE = _phrase_regex(STRONG_PHRASES)
_QWORD_RE = _phrase_regex(QUESTION_WORDS)
_NUMWORD_RE = _phrase_regex(NUMBER_WORDS)


@dataclass
class Unit:
    """A sentence-like span of speech: the smallest piece a clip may start/end on."""

    start: float
    end: float
    text: str
    words: list[dict] = field(default_factory=list)
    # True when the unit ends a sentence (terminal punctuation, or a clear pause
    # after it). Clips may only start after, and end on, such units.
    ends_sentence: bool = True


@dataclass
class Clip:
    """A scored clip window."""

    start: float
    end: float
    score: int
    title: str
    reason: str
    text: str
    words: list[dict] = field(default_factory=list)
    features: dict = field(default_factory=dict)
    source: str = "heuristic"

    @property
    def duration(self) -> float:
        return self.end - self.start


# --------------------------------------------------------------------------- units

PAUSE_BREAK = 1.0  # a silence this long (s) counts as a sentence boundary


def build_units(segments: list[dict]) -> list[Unit]:
    """Split the transcript into sentence units using word timestamps.

    Words from all Whisper segments are treated as one continuous stream,
    because Whisper's segment edges often fall mid-sentence. A unit ends at
    terminal punctuation or a clear pause. Very long sentences are cut further
    at word boundaries (prefer commas) so a clip can always fit inside the max
    length; those cut points are marked ``ends_sentence=False``.
    """
    stream: list[dict] = []
    units: list[Unit] = []
    for seg in segments:
        words = [w for w in seg.get("words") or [] if w.get("word", "").strip()]
        if words:
            stream.extend(words)
        elif seg.get("text", "").strip():  # no word timings: keep the segment as one unit
            units.append(Unit(seg["start"], seg["end"], seg["text"].strip(), []))
    stream.sort(key=lambda w: w["start"])

    current: list[dict] = []
    for k, w in enumerate(stream):
        current.append(w)
        token = w["word"].strip()
        nxt = stream[k + 1] if k + 1 < len(stream) else None
        pause = (nxt["start"] - w["end"]) if nxt else PAUSE_BREAK
        dur = w["end"] - current[0]["start"]
        if _SENTENCE_END.search(token) or pause >= PAUSE_BREAK:
            units.append(_unit_from_words(current, ends_sentence=True))
            current = []
        elif (dur >= 8 and token.endswith((",", ";", ":"))) or dur >= 15:
            units.append(_unit_from_words(current, ends_sentence=False))
            current = []
    if current:
        units.append(_unit_from_words(current, ends_sentence=True))
    units.sort(key=lambda u: u.start)
    return units


def _unit_from_words(words: list[dict], ends_sentence: bool = True) -> Unit:
    text = "".join(w["word"] for w in words).strip()
    return Unit(words[0]["start"], words[-1]["end"], text, list(words), ends_sentence)


def padded_bounds(units: list[Unit], i: int, j: int, duration: float, pad: float = PAD) -> tuple[float, float]:
    """Clip bounds for units i..j with padding that never reaches into a neighbouring word."""
    start = units[i].start - pad
    if i > 0:
        start = max(start, (units[i - 1].end + units[i].start) / 2)
    end = units[j].end + pad
    if j + 1 < len(units):
        end = min(end, (units[j].end + units[j + 1].start) / 2)
    return max(0.0, start), min(duration, end) if duration > 0 else end


# --------------------------------------------------------------------------- features

def hook_points(text: str, first_sentence: str = "") -> tuple[float, list[str]]:
    """Return a raw hook value and the matched strong phrases."""
    strong = [m.group(0).lower() for m in _STRONG_RE.finditer(text)]
    value = 1.0 * len(strong)
    value += 0.6 * text.count("?") + 0.4 * text.count("!")
    value += 0.3 * (len(_NUMBER_RE.findall(text)) + len(_NUMWORD_RE.findall(text)))
    value += 0.2 * len(_QWORD_RE.findall(text))
    if first_sentence:  # a hook in the opening line matters most
        value += 0.8 * len(_STRONG_RE.findall(first_sentence)) + 0.5 * first_sentence.count("?")
    return value, strong


class Scorer:
    """Computes features for windows and maps them to a 0-100 score."""

    def __init__(self, energy: np.ndarray, hop: float = ENERGY_HOP) -> None:
        self.energy = energy
        self.hop = hop
        voiced = energy[energy > -60] if energy.size else energy
        self.global_db = float(np.mean(voiced)) if voiced.size else -40.0
        self.pace_lo, self.pace_hi = 1.5, 3.5
        self.energy_lo, self.energy_hi = -4.0, 4.0

    def window_energy(self, start: float, end: float) -> float:
        if not self.energy.size:
            return 0.0
        a = int(start / self.hop)
        b = max(a + 1, int(math.ceil(end / self.hop)))
        frames = self.energy[a:b]
        if not frames.size:
            return 0.0
        return float(np.mean(frames)) - self.global_db

    def raw_features(self, start: float, end: float, words: list[dict], first_sentence: str) -> dict:
        dur = max(0.1, end - start)
        text = "".join(w["word"] for w in words)
        n_words = len(words)
        gaps = 0.0
        prev_end = start
        for w in words:
            gap = w["start"] - prev_end
            if gap > 0.7:
                gaps += gap - 0.7
            prev_end = max(prev_end, w["end"])
        tail = end - prev_end
        if tail > 0.7:
            gaps += tail - 0.7
        hook, phrases = hook_points(text, first_sentence)
        return {
            "energy_db": self.window_energy(start, end),
            "pace": n_words / dur,
            "hook": hook,
            "phrases": phrases,
            "silence_ratio": min(1.0, gaps / dur),
            "questions": text.count("?"),
        }

    def calibrate(self, features: list[dict]) -> None:
        """Set normalisation ranges from all candidates (robust 10th-90th percentiles)."""
        if len(features) < 4:
            return
        e = np.array([f["energy_db"] for f in features])
        p = np.array([f["pace"] for f in features])
        self.energy_lo, self.energy_hi = float(np.percentile(e, 10)), float(np.percentile(e, 90))
        self.pace_lo, self.pace_hi = float(np.percentile(p, 10)), float(np.percentile(p, 90))
        if self.energy_hi - self.energy_lo < 1.0:
            mid = (self.energy_hi + self.energy_lo) / 2
            self.energy_lo, self.energy_hi = mid - 1.5, mid + 1.5
        if self.pace_hi - self.pace_lo < 0.3:
            mid = (self.pace_hi + self.pace_lo) / 2
            self.pace_lo, self.pace_hi = mid - 0.5, mid + 0.5

    @staticmethod
    def _norm(value: float, lo: float, hi: float) -> float:
        return float(min(1.0, max(0.0, (value - lo) / (hi - lo)))) if hi > lo else 0.5

    def score(self, f: dict) -> tuple[int, dict]:
        e = self._norm(f["energy_db"], self.energy_lo, self.energy_hi)
        p = self._norm(f["pace"], self.pace_lo, self.pace_hi)
        h = 1.0 - math.exp(-f["hook"] / 2.0)
        s = min(1.0, f["silence_ratio"] / 0.25)
        raw = 0.35 * e + 0.20 * p + 0.30 * h + 0.15 * (1.0 - s)
        parts = {"energy": round(e, 3), "pace": round(p, 3), "hook": round(h, 3), "silence": round(s, 3)}
        return int(round(100 * raw)), parts


def _reason(f: dict, parts: dict) -> str:
    bits = []
    if f["phrases"]:
        uniq = list(dict.fromkeys(f["phrases"]))[:2]
        bits.append("hook phrase " + ", ".join(f"'{p}'" for p in uniq))
    if f["questions"]:
        bits.append("asks a question")
    if parts["energy"] >= 0.6:
        bits.append("louder / more energetic than average")
    if parts["pace"] >= 0.6:
        bits.append(f"fast pace ({f['pace']:.1f} words/s)")
    if parts["silence"] <= 0.1:
        bits.append("no dead air")
    return ("; ".join(bits) or "steady, well-paced speech").capitalize()


def make_title(text: str, phrases: list[str], limit: int = 70) -> str:
    """Pick a short title: the sentence with a hook phrase, else the opening sentence."""
    sentences = [s.strip() for s in re.split(r"(?<=[.?!।])\s+", text) if s.strip()]
    if not sentences:
        return "Highlight"
    chosen = sentences[0]
    for s in sentences:
        if any(p in s.lower() for p in phrases) or "?" in s:
            chosen = s
            break
    chosen = chosen.strip(" \"'")
    if len(chosen) > limit:
        chosen = chosen[:limit].rsplit(" ", 1)[0].rstrip(",;:") + "..."
    return chosen


# --------------------------------------------------------------------------- candidates

def words_in(units: list[Unit], i: int, j: int) -> list[dict]:
    out: list[dict] = []
    for u in units[i: j + 1]:
        out.extend(u.words)
    return out


def score_span(units: list[Unit], i: int, j: int, scorer: Scorer, duration: float, source: str = "heuristic") -> Clip:
    start, end = padded_bounds(units, i, j, duration)
    words = words_in(units, i, j)
    text = " ".join(u.text for u in units[i: j + 1])
    f = scorer.raw_features(start, end, words, units[i].text)
    score, parts = scorer.score(f)
    return Clip(start=round(start, 3), end=round(end, 3), score=score, title=make_title(text, f["phrases"]),
                reason=_reason(f, parts), text=text, words=words, features=parts, source=source)


def build_candidates(units: list[Unit], min_len: float, max_len: float, duration: float) -> list[tuple[int, int]]:
    """Unit spans (i, j) whose padded length lies within [min_len, max_len].

    Spans must start at the beginning of a sentence and end at the end of one.
    If that leaves nothing (e.g. one endless run-on sentence), any span is allowed.
    """
    def collect(whole_sentences: bool) -> list[tuple[int, int]]:
        spans: list[tuple[int, int]] = []
        n = len(units)
        for i in range(n):
            if whole_sentences and i > 0 and not units[i - 1].ends_sentence:
                continue
            for j in range(i, n):
                s, e = padded_bounds(units, i, j, duration)
                length = e - s
                if length > max_len:
                    break
                if length >= min_len and (not whole_sentences or units[j].ends_sentence):
                    spans.append((i, j))
        return spans

    return collect(True) or collect(False)


def select_top(clips: list[Clip], n: int) -> list[Clip]:
    """Greedy pick of the best non-overlapping clips, returned best first."""
    chosen: list[Clip] = []
    for c in sorted(clips, key=lambda c: c.score, reverse=True):
        if all(c.end <= o.start or c.start >= o.end for o in chosen):
            chosen.append(c)
            if len(chosen) >= n:
                break
    return chosen


def heuristic_clips(
    units: list[Unit], scorer: Scorer, n: int, min_len: float, max_len: float, duration: float
) -> list[Clip]:
    """Score every candidate span and return the top ``n`` non-overlapping clips."""
    spans = build_candidates(units, min_len, max_len, duration)
    if not spans and units:
        # Speech is shorter than min_len (or one sentence is huge): use what fits.
        j = 0
        while j + 1 < len(units) and padded_bounds(units, 0, j + 1, duration)[1] - padded_bounds(units, 0, j + 1, duration)[0] <= max_len:
            j += 1
        spans = [(0, j)]
    feats = []
    for i, j in spans:
        s, e = padded_bounds(units, i, j, duration)
        feats.append(scorer.raw_features(s, e, words_in(units, i, j), units[i].text))
    scorer.calibrate(feats)
    clips = [score_span(units, i, j, scorer, duration) for i, j in spans]
    return select_top(clips, n)


def energy_only_clips(energy: np.ndarray, n: int, min_len: float, max_len: float, duration: float,
                      hop: float = ENERGY_HOP) -> list[Clip]:
    """Fallback for videos with little or no speech: pick the loudest/most dynamic windows."""
    length = min(max_len, max(min_len, (min_len + max_len) / 2))
    if duration <= length:
        return [Clip(0.0, round(duration, 3), 50, "Full video", "Video is shorter than the clip length", "")]
    if not energy.size:  # no audio at all: spread clips evenly over the video
        count = max(1, min(n, int(duration // length)))
        gap = (duration - count * length) / count
        clips = []
        for k in range(count):
            start = k * (length + gap) + gap / 2
            mm, ss = divmod(int(start), 60)
            clips.append(Clip(round(start, 3), round(start + length, 3), 50, f"Moment at {mm}:{ss:02d}",
                              "No audio track - clips spread evenly across the video", ""))
        return clips
    frames = max(1, int(length / hop))
    step = max(1, int(1.0 / hop))
    windows: list[tuple[float, int]] = []
    for a in range(0, max(1, len(energy) - frames), step):
        seg = energy[a: a + frames]
        windows.append((float(np.mean(seg) + 0.5 * np.std(seg)), a))
    vals = np.array([w[0] for w in windows])
    lo, hi = float(np.percentile(vals, 5)), float(vals.max())
    clips: list[Clip] = []
    for value, a in windows:
        start = a * hop
        # Snap the start to the quietest frame within 1.5 s so we cut on a lull.
        lo_f, hi_f = max(0, a - int(1.5 / hop)), min(len(energy), a + int(1.5 / hop))
        quiet = lo_f + int(np.argmin(energy[lo_f:hi_f])) if hi_f > lo_f else a
        start = quiet * hop
        end = min(duration, start + length)
        score = int(round(100 * min(1.0, max(0.0, (value - lo) / (hi - lo))))) if hi > lo else 50
        mm, ss = divmod(int(round(start)), 60)
        clips.append(Clip(round(start, 3), round(end, 3), score, f"High-energy moment at {mm}:{ss:02d}",
                          "Little or no speech found - picked by audio energy", ""))
    return select_top(clips, n)
