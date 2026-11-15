"""Optional AI clip picker using Ollama (local server or Ollama Cloud).

API (docs.ollama.com):
* POST {base}/api/chat  with {"model", "messages", "stream": false[, "format": "json"]}
  -> answer text in response["message"]["content"]
* GET  {base}/api/tags  -> installed models (used for "Test connection")
* Cloud: base https://ollama.com, header "Authorization: Bearer <key>".
  Cloud models do not support structured outputs, so "format" is only sent locally.

Any failure raises ``AIPickError``; the caller falls back to the heuristic.
"""

from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass

import requests

from .errors import JobCancelled
from .scoring import Clip, Scorer, Unit, padded_bounds, score_span

DEFAULT_LOCAL_URL = "http://localhost:11434"
DEFAULT_CLOUD_URL = "https://ollama.com"
CHUNK_CHARS = 9000  # transcript characters per request (keeps prompts within small context windows)


class AIPickError(Exception):
    """The AI picker could not produce usable clips."""


@dataclass
class OllamaSettings:
    mode: str = "local"  # "local" or "cloud"
    base_url: str = DEFAULT_LOCAL_URL
    model: str = "llama3.2"
    api_key: str = ""
    timeout: float = 180.0

    @classmethod
    def from_dict(cls, d: dict | None) -> "OllamaSettings":
        d = d or {}
        mode = "cloud" if d.get("mode") == "cloud" else "local"
        default_url = DEFAULT_CLOUD_URL if mode == "cloud" else DEFAULT_LOCAL_URL
        return cls(
            mode=mode,
            base_url=(str(d.get("base_url") or default_url)).rstrip("/"),
            model=str(d.get("model") or "llama3.2"),
            api_key=str(d.get("api_key") or ""),
            timeout=float(d.get("timeout") or 180.0),
        )

    def headers(self) -> dict:
        h = {"Content-Type": "application/json"}
        if self.mode == "cloud" and self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h


def test_connection(settings: OllamaSettings) -> dict:
    """Return {"ok": bool, "message": str, "models": [...]} without raising."""
    try:
        r = requests.get(f"{settings.base_url}/api/tags", headers=settings.headers(), timeout=10)
        if r.status_code == 401:
            return {"ok": False, "message": "Unauthorized - check the API key.", "models": []}
        r.raise_for_status()
        models = [m.get("name", "") for m in r.json().get("models", [])]
        found = any(m == settings.model or m.split(":")[0] == settings.model for m in models)
        msg = f"Connected. {len(models)} model(s) available."
        if models and not found:
            msg += f" Model '{settings.model}' was not found in the list."
        return {"ok": True, "message": msg, "models": models[:50]}
    except requests.RequestException as exc:
        return {"ok": False, "message": f"Could not reach Ollama at {settings.base_url}: {exc.__class__.__name__}", "models": []}
    except ValueError:
        return {"ok": False, "message": "Ollama answered with something that is not JSON.", "models": []}


def _fmt(t: float) -> str:
    return f"{t:.1f}"


def _chunks(units: list[Unit]) -> list[list[Unit]]:
    chunks: list[list[Unit]] = [[]]
    size = 0
    for u in units:
        line_len = len(u.text) + 16
        if size + line_len > CHUNK_CHARS and chunks[-1]:
            chunks.append([])
            size = 0
        chunks[-1].append(u)
        size += line_len
    return [c for c in chunks if c]


def _prompt(chunk: list[Unit], n: int, min_len: float, max_len: float) -> str:
    lines = "\n".join(f"[{_fmt(u.start)}-{_fmt(u.end)}] {u.text}" for u in chunk)
    return (
        f"Below is a timestamped transcript (times in seconds). Find the {n} most engaging, "
        f"self-contained moments that would work as short social-media clips. Each clip must be "
        f"between {min_len:.0f} and {max_len:.0f} seconds long, start at the beginning of a sentence "
        f"and end at the end of a sentence, and make sense without extra context. Prefer strong hooks, "
        f"surprising facts, emotion, humour, clear advice or a story with a payoff.\n\n"
        f"Reply with ONLY strict JSON, no markdown, in exactly this shape:\n"
        f'{{"clips": [{{"start": 12.3, "end": 45.6, "title": "short catchy title", '
        f'"reason": "why this is engaging"}}]}}\n\nTranscript:\n{lines}'
    )


def _extract_json(text: str) -> list[dict]:
    """Parse the model reply leniently (code fences, leading prose) but validate strictly."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
    # Some reasoning models wrap their thinking in <think> tags.
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    data = None
    try:
        data = json.loads(text)
    except ValueError:
        m = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
        if m:
            try:
                data = json.loads(m.group(1))
            except ValueError:
                data = None
    if data is None:
        raise AIPickError("The AI reply was not valid JSON.")
    if isinstance(data, dict):
        data = data.get("clips") or data.get("moments") or data.get("highlights") or []
    if not isinstance(data, list):
        raise AIPickError("The AI reply did not contain a list of clips.")
    out = []
    for item in data:
        if not isinstance(item, dict):
            continue
        try:
            start, end = float(item["start"]), float(item["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if end > start >= 0:
            out.append({"start": start, "end": end, "title": str(item.get("title") or "").strip()[:100],
                        "reason": str(item.get("reason") or "").strip()[:300]})
    return out


def _chat(settings: OllamaSettings, prompt: str, cancel_event: threading.Event) -> str:
    body: dict = {
        "model": settings.model,
        "messages": [
            {"role": "system", "content": "You are a video editor who finds viral short-form clips. You answer only in JSON."},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "options": {"temperature": 0.2},
    }
    if settings.mode == "local":
        body["format"] = "json"
    if cancel_event.is_set():
        raise JobCancelled()
    try:
        r = requests.post(f"{settings.base_url}/api/chat", json=body, headers=settings.headers(),
                          timeout=settings.timeout)
    except requests.Timeout as exc:
        raise AIPickError("Ollama timed out.") from exc
    except requests.RequestException as exc:
        raise AIPickError(f"Ollama is unreachable at {settings.base_url}.") from exc
    if r.status_code == 401:
        raise AIPickError("Ollama rejected the API key (401).")
    if r.status_code == 404:
        raise AIPickError(f"Ollama model '{settings.model}' was not found (404). Pull it first: ollama pull {settings.model}")
    if r.status_code >= 400:
        raise AIPickError(f"Ollama returned HTTP {r.status_code}.")
    try:
        return str(r.json()["message"]["content"])
    except (ValueError, KeyError, TypeError) as exc:
        raise AIPickError("Unexpected response format from Ollama.") from exc


def _snap(units: list[Unit], start: float, end: float, min_len: float, max_len: float,
          duration: float) -> tuple[int, int] | None:
    """Snap AI times to unit boundaries and clamp to the allowed length."""
    if not units:
        return None
    # Valid boundaries: clips start right after a sentence end and end on one.
    starts = [k for k in range(len(units)) if k == 0 or units[k - 1].ends_sentence] or list(range(len(units)))
    ends = [k for k in range(len(units)) if units[k].ends_sentence] or list(range(len(units)))

    def length(a: int, b: int) -> float:
        s, e = padded_bounds(units, a, b, duration)
        return e - s

    i = min(starts, key=lambda k: abs(units[k].start - start))
    valid_ends = [k for k in ends if k >= i]
    if not valid_ends:
        return None
    j = min(valid_ends, key=lambda k: abs(units[k].end - end))
    pos = valid_ends.index(j)
    while pos > 0 and length(i, valid_ends[pos]) > max_len:  # too long: end earlier
        pos -= 1
    while length(i, valid_ends[pos]) < min_len and pos + 1 < len(valid_ends) \
            and length(i, valid_ends[pos + 1]) <= max_len:  # too short: end later
        pos += 1
    j = valid_ends[pos]
    s_pos = starts.index(i)
    while length(i, j) < min_len and s_pos > 0 and length(starts[s_pos - 1], j) <= max_len:  # start earlier
        s_pos -= 1
        i = starts[s_pos]
    if length(i, j) > max_len + 0.01:
        return None
    return i, j


def pick_clips(
    units: list[Unit], scorer: Scorer, settings: OllamaSettings, n: int, min_len: float, max_len: float,
    duration: float, cancel_event: threading.Event, progress=None,
) -> list[Clip]:
    """Ask Ollama for clips. Raises AIPickError on any problem so the caller can fall back."""
    if not units:
        raise AIPickError("No transcript to send to the AI.")
    if settings.mode == "cloud" and not settings.api_key:
        raise AIPickError("Ollama Cloud needs an API key.")
    chunks = _chunks(units)
    per_chunk = max(2, min(n, (n * 2) // len(chunks) + 1)) if len(chunks) > 1 else n
    proposals: list[dict] = []
    for idx, chunk in enumerate(chunks):
        if progress:
            progress(idx / len(chunks) * 100, f"Asking {settings.model} (part {idx + 1} of {len(chunks)})")
        reply = _chat(settings, _prompt(chunk, per_chunk, min_len, max_len), cancel_event)
        proposals.extend(_extract_json(reply))
    if not proposals:
        raise AIPickError("The AI did not return any usable clips.")

    clips: list[Clip] = []
    seen: set[tuple[int, int]] = set()
    for rank, p in enumerate(proposals):
        span = _snap(units, p["start"], p["end"], min_len, max_len, duration)
        if span is None or span in seen:
            continue
        seen.add(span)
        clip = score_span(units, span[0], span[1], scorer, duration, source="ai")
        if p["title"]:
            clip.title = p["title"]
        if p["reason"]:
            clip.reason = p["reason"]
        clip.features["ai_rank"] = rank
        clips.append(clip)
    # Keep the AI's order of preference; drop overlaps.
    chosen: list[Clip] = []
    for c in clips:
        if all(c.end <= o.start or c.start >= o.end for o in chosen):
            chosen.append(c)
        if len(chosen) >= n:
            break
    if not chosen:
        raise AIPickError("None of the AI's suggestions fit the clip length limits.")
    return chosen
