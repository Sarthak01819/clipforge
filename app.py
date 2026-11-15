"""ClipForge web server (FastAPI) - serves the UI and runs pipeline jobs in the background.

Run:  .venv\\Scripts\\python app.py   (or run.bat)  ->  http://localhost:7860
"""

from __future__ import annotations

import json
import logging
import shutil
import threading
import time
import uuid
import zipfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import unquote

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from pipeline import JobCancelled, JobOptions, PipelineError, Source, run_job
from pipeline.ai_picker import DEFAULT_CLOUD_URL, DEFAULT_LOCAL_URL, OllamaSettings, test_connection
from pipeline.ffmpeg_utils import get_capabilities
from pipeline.ingest import check_upload_extension, is_probably_url, sanitize_filename
from pipeline.transcribe import WHISPER_MODELS, _cuda_available

ROOT = Path(__file__).resolve().parent
JOBS_DIR = ROOT / "jobs"
MODELS_DIR = ROOT / "models"
STATIC_DIR = ROOT / "static"
CONFIG_PATH = ROOT / "config.json"
HOST, PORT = "127.0.0.1", 7860

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("clipforge")
JOBS_DIR.mkdir(exist_ok=True)

app = FastAPI(title="ClipForge")
executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="job")  # one heavy job at a time


# --------------------------------------------------------------------------- config

def load_config() -> dict:
    try:
        return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save_config(cfg: dict) -> None:
    tmp = CONFIG_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    tmp.replace(CONFIG_PATH)


def ollama_config() -> dict:
    o = load_config().get("ollama") or {}
    mode = "cloud" if o.get("mode") == "cloud" else "local"
    return {
        "mode": mode,
        "base_url": o.get("base_url") or (DEFAULT_CLOUD_URL if mode == "cloud" else DEFAULT_LOCAL_URL),
        "model": o.get("model") or "llama3.2",
        "api_key": o.get("api_key") or "",
        "timeout": o.get("timeout") or 180,
    }


# --------------------------------------------------------------------------- jobs

@dataclass
class Job:
    id: str
    dir: Path
    state: str = "created"  # created | queued | running | done | error | cancelled
    stage: str = ""
    stage_pct: float = 0.0
    overall: float = 0.0
    message: str = ""
    error: str | None = None
    detail: str | None = None
    result: dict | None = None
    created: float = field(default_factory=time.time)
    source_label: str = ""
    cancel_event: threading.Event = field(default_factory=threading.Event)
    options: dict = field(default_factory=dict)
    url: str | None = None

    def public(self) -> dict[str, Any]:
        return {
            "id": self.id, "state": self.state, "stage": self.stage, "stage_pct": round(self.stage_pct, 1),
            "overall": round(self.overall, 1), "message": self.message, "error": self.error,
            "detail": self.detail, "result": self.result, "created": self.created, "source": self.source_label,
        }


jobs: dict[str, Job] = {}
jobs_lock = threading.Lock()


def _load_finished_jobs() -> None:
    """Re-list jobs that finished in a previous server run."""
    for d in sorted(JOBS_DIR.iterdir()) if JOBS_DIR.exists() else []:
        res = d / "result.json"
        if d.is_dir() and res.is_file():
            try:
                result = json.loads(res.read_text(encoding="utf-8"))
            except ValueError:
                continue
            jobs[d.name] = Job(id=d.name, dir=d, state="done", stage="Done", overall=100, result=result,
                               created=res.stat().st_mtime, source_label=result.get("title", ""))


_load_finished_jobs()


def get_job(job_id: str) -> Job:
    with jobs_lock:
        job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return job


def _worker(job: Job, source: Source) -> None:
    if job.cancel_event.is_set():
        job.state, job.message = "cancelled", "Cancelled"
        return
    job.state = "running"

    def report(stage: str, pct: float, msg: str, overall: float) -> None:
        job.stage, job.stage_pct, job.message, job.overall = stage, pct, msg, overall

    try:
        opts = JobOptions.from_dict(job.options)
        job.result = run_job(job.dir, source, opts, report, job.cancel_event, MODELS_DIR)
        job.state, job.stage, job.overall, job.message = "done", "Done", 100.0, "Finished"
    except JobCancelled:
        job.state, job.stage, job.message = "cancelled", "Cancelled", "The job was cancelled."
    except PipelineError as exc:
        log.warning("Job %s failed: %s | %s", job.id, exc.message, exc.detail)
        job.state, job.error, job.detail = "error", exc.message, exc.detail
    except Exception as exc:  # noqa: BLE001 - last line of defence, keep the server alive
        log.exception("Job %s crashed", job.id)
        job.state, job.error, job.detail = "error", "Something went wrong while processing the video.", repr(exc)


def _start(job: Job, source: Source) -> None:
    job.state, job.stage, job.message = "queued", "Queued", "Waiting for the previous job to finish"
    executor.submit(_worker, job, source)


# --------------------------------------------------------------------------- API

@app.get("/api/system")
def system_info() -> dict:
    caps = get_capabilities()
    return {
        "ffmpeg": caps.path, "ffmpeg_version": caps.version, "has_ass": caps.has_ass,
        "has_nvenc": caps.has_nvenc, "cuda": _cuda_available(), "warnings": caps.warnings,
        "whisper_models": list(WHISPER_MODELS),
    }


@app.post("/api/jobs")
async def create_job(payload: dict) -> dict:
    """Create a job. For URLs it starts at once; for uploads it waits for PUT /upload."""
    source_type = payload.get("source")
    options = dict(payload.get("options") or {})
    if options.get("use_ai"):
        options["ollama"] = ollama_config()  # settings (incl. key) come from config.json, not the browser
    job_id = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
    job_dir = JOBS_DIR / job_id
    if source_type == "url":
        url = str(payload.get("url") or "").strip()
        if not is_probably_url(url):
            raise HTTPException(400, "That does not look like a valid link. It should start with http:// or https://.")
        job_dir.mkdir(parents=True)
        job = Job(id=job_id, dir=job_dir, options=options, url=url, source_label=url)
        with jobs_lock:
            jobs[job_id] = job
        _start(job, Source(url=url))
    elif source_type == "upload":
        filename = sanitize_filename(str(payload.get("filename") or "video.mp4"))
        try:
            check_upload_extension(filename)
        except PipelineError as exc:
            raise HTTPException(400, exc.message) from exc
        job_dir.mkdir(parents=True)
        job = Job(id=job_id, dir=job_dir, options=options, source_label=filename)
        with jobs_lock:
            jobs[job_id] = job
    else:
        raise HTTPException(400, "Choose a file to upload or paste a link.")
    return job.public()


@app.put("/api/jobs/{job_id}/upload")
async def upload(job_id: str, request: Request) -> dict:
    """Stream the request body straight to disk (never held in memory), then start the job."""
    job = get_job(job_id)
    if job.state != "created":
        raise HTTPException(409, "This job already has its video.")
    filename = sanitize_filename(unquote(request.headers.get("x-filename", "")) or job.source_label)
    try:
        check_upload_extension(filename)
    except PipelineError as exc:
        raise HTTPException(400, exc.message) from exc
    dest = job.dir / f"upload_{filename}"
    size = 0
    try:
        with open(dest, "wb") as fh:
            async for chunk in request.stream():
                if job.cancel_event.is_set():
                    raise HTTPException(499, "Upload cancelled")
                fh.write(chunk)
                size += len(chunk)
    except Exception:
        dest.unlink(missing_ok=True)
        job.state, job.error = "error", "The upload was interrupted."
        raise
    if size == 0:
        dest.unlink(missing_ok=True)
        job.state, job.error = "error", "The uploaded file is empty."
        raise HTTPException(400, "The uploaded file is empty.")
    job.source_label = filename
    _start(job, Source(file=dest, original_name=Path(filename).stem))
    return job.public()


@app.get("/api/jobs/{job_id}/status")
def status(job_id: str) -> dict:
    return get_job(job_id).public()


@app.get("/api/jobs")
def list_jobs() -> list[dict]:
    with jobs_lock:
        items = sorted(jobs.values(), key=lambda j: j.created, reverse=True)
    return [{"id": j.id, "state": j.state, "source": j.source_label, "created": j.created,
             "clips": len((j.result or {}).get("clips", []))} for j in items]


@app.post("/api/jobs/{job_id}/cancel")
def cancel(job_id: str) -> dict:
    job = get_job(job_id)
    job.cancel_event.set()
    if job.state in ("created", "queued"):
        job.state, job.stage, job.message = "cancelled", "Cancelled", "The job was cancelled."
    return job.public()


@app.delete("/api/jobs")
def clear_jobs() -> dict:
    """Delete every job folder that is not currently queued or running."""
    removed = 0
    with jobs_lock:
        active = {j.id for j in jobs.values() if j.state in ("queued", "running", "created")}
        for d in list(JOBS_DIR.iterdir()):
            if d.is_dir() and d.name not in active:
                shutil.rmtree(d, ignore_errors=True)
                jobs.pop(d.name, None)
                removed += 1
    return {"removed": removed}


def _safe_file(job: Job, name: str) -> Path:
    safe = sanitize_filename(name)
    if safe != name:
        raise HTTPException(404, "File not found")
    path = (job.dir / safe).resolve()
    if path.parent != job.dir.resolve() or not path.is_file():
        raise HTTPException(404, "File not found")
    return path


@app.get("/api/jobs/{job_id}/files/{name}")
def get_file(job_id: str, name: str, download: int = 0):
    job = get_job(job_id)
    path = _safe_file(job, name)
    media = {".mp4": "video/mp4", ".srt": "application/x-subrip", ".json": "application/json"}.get(path.suffix, "application/octet-stream")
    if download:
        return FileResponse(path, media_type=media, filename=path.name)
    return FileResponse(path, media_type=media)


@app.get("/api/jobs/{job_id}/zip")
def get_zip(job_id: str):
    job = get_job(job_id)
    if job.state != "done" or not job.result:
        raise HTTPException(409, "The clips are not ready yet.")
    zip_path = job.dir / "clips.zip"
    files = []
    for c in job.result.get("clips", []):
        for key in ("file", "srt"):
            if c.get(key) and (job.dir / c[key]).is_file():
                files.append(job.dir / c[key])
    newest = max((f.stat().st_mtime for f in files), default=0)
    if not zip_path.exists() or zip_path.stat().st_mtime < newest:
        tmp = job.dir / "clips.zip.tmp"
        with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as zf:
            for f in files:
                zf.write(f, arcname=f.name)
        tmp.replace(zip_path)
    stem = sanitize_filename(job.result.get("title") or "clips", default="clips").rsplit(".", 1)[0]
    return FileResponse(zip_path, media_type="application/zip", filename=f"{stem}_clips.zip")


@app.get("/api/config")
def get_config() -> dict:
    o = ollama_config()
    return {"ollama": {"mode": o["mode"], "base_url": o["base_url"], "model": o["model"],
                       "has_api_key": bool(o["api_key"])}}


@app.post("/api/config")
def set_config(payload: dict) -> dict:
    incoming = payload.get("ollama") or {}
    cfg = load_config()
    current = ollama_config()
    mode = "cloud" if incoming.get("mode") == "cloud" else "local"
    new = {
        "mode": mode,
        "base_url": str(incoming.get("base_url") or (DEFAULT_CLOUD_URL if mode == "cloud" else DEFAULT_LOCAL_URL)).strip().rstrip("/"),
        "model": str(incoming.get("model") or current["model"]).strip(),
        "api_key": current["api_key"],
        "timeout": current["timeout"],
    }
    if not new["base_url"].startswith(("http://", "https://")):
        raise HTTPException(400, "The base URL must start with http:// or https://")
    if "api_key" in incoming and incoming["api_key"] is not None:
        new["api_key"] = str(incoming["api_key"]).strip()  # empty string clears the key
    cfg["ollama"] = new
    save_config(cfg)
    return get_config()


@app.post("/api/ollama/test")
def ollama_test() -> dict:
    return test_connection(OllamaSettings.from_dict(ollama_config()))


@app.exception_handler(HTTPException)
async def http_error(_: Request, exc: HTTPException):
    return JSONResponse({"error": exc.detail}, status_code=exc.status_code)


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html", headers={"Cache-Control": "no-cache"})


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# Probe ffmpeg (filters, NVENC) in the background so the first page load is fast.
threading.Thread(target=get_capabilities, daemon=True).start()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=HOST, port=PORT, log_level="info")
