# ClipForge

ClipForge is a local tool that turns a long video into short clips. Give it a file or a link, and it:

1. downloads or saves the video,
2. transcribes it with Whisper (word-level timestamps),
3. finds the most engaging moments,
4. cuts them into separate MP4 clips (original aspect or vertical 9:16),
5. optionally burns in captions with word-by-word highlighting,
6. lets you preview each clip, download it, or download everything as a zip.

Everything runs on your own computer. No paid services, no account and no API key are needed for the core features.

> **Only clip videos you own or have the rights to use.** Downloading or re-uploading other people's content can break copyright law and the site's terms of service.

## Requirements

- Windows 10 or 11
- Python 3.10 or newer ([python.org](https://www.python.org/downloads/); tick **Add python.exe to PATH**)
- About 2 GB of free disk space for packages and Whisper models
- Internet access the first time, to install packages and download the Whisper model you choose
- FFmpeg is optional. If `ffmpeg` is on your PATH it is used. If not, the copy bundled with the `imageio-ffmpeg` package is used automatically.

A GPU is not required. If you have an NVIDIA GPU:
- **Rendering** uses the NVENC hardware encoder automatically when it is available.
- **Transcription** uses the GPU when the CUDA 12 libraries are installed, which `setup.bat` does for you (see [GPU transcription](#gpu-transcription)). Otherwise it runs on the CPU with int8 quantisation.

## Setup (CMD)

```bat
cd path\to\video-clipper
setup.bat
```

`setup.bat` creates a virtual environment in `.venv` and installs everything from `requirements.txt` into it, plus the GPU libraries from `requirements-gpu.txt` if an NVIDIA GPU is found. Run it again at any time to update packages, including yt-dlp. Sites change often, so updating yt-dlp fixes most download errors.

## Run

```bat
run.bat
```

`run.bat` activates the venv, starts the server at **http://localhost:7860** and opens your browser. Press `Ctrl+C` in the window to stop the server.

You can also start it manually:

```bat
.venv\Scripts\activate
python app.py
```

## Using it

1. Drop a video file on the upload box, or paste a link (YouTube, Vimeo, X and [any site yt-dlp supports](https://github.com/yt-dlp/yt-dlp/blob/master/supportedsites.md)). Links are downloaded at the best quality up to 1080p.
2. Choose your options:
   - **Number of clips**, **min/max length** (defaults: 5 clips, 20 to 60 seconds)
   - **Format**: original aspect ratio, or vertical 9:16 (centre crop to 1080x1920)
   - **Captions**: on/off, size (small, medium or large) and position (bottom or middle)
   - **Whisper model**: tiny, base (default), small or medium. Larger models are more accurate but slower, and each model is downloaded once into `models/`.
   - **Language**: auto-detect, or force one such as English or Hindi
3. Click **Find clips**. Progress shows each stage: Downloading, Extracting audio, Transcribing with a percentage, Finding highlights, and Rendering clip N of M. **Cancel** stops the job.
4. Each clip card has a player, title, time range, score, the reason it was picked, the transcript, an **MP4** download and an **SRT** download. **Download all (zip)** bundles every clip and its subtitles.
5. **Clear old jobs** deletes all finished, failed and cancelled job folders under `jobs/`.

## How clips are chosen

Candidate clips start and end on sentence boundaries, never in the middle of a word, with about 0.3 s of padding. Each candidate gets a score from 0 to 100:

| Signal | Weight | Meaning |
|---|---|---|
| Loudness | 35% | Audio energy compared with the video's average |
| Pace | 20% | Words per second |
| Hooks | 30% | Questions, exclamations, numbers and strong phrases ("the secret", "nobody tells you", "here's why", "biggest mistake", "सबसे बड़ी गलती", "कोई नहीं बताता", ...). A hook in the opening line counts extra. |
| No dead air | 15% | Long silences pull the score down |

The top N clips that don't overlap are kept. If a video has little or no speech, clips are picked by audio energy alone and captions are skipped.

## Optional: AI clip-picker with Ollama

This is off by default. When enabled, the timestamped transcript is sent to an Ollama model (in chunks for long videos). The model returns the most engaging self-contained moments as strict JSON with `start`, `end`, `title` and `reason`. ClipForge validates the reply, snaps the times to sentence boundaries and clamps each clip to your min/max length.

If Ollama is unreachable, times out or returns bad JSON, ClipForge falls back to the built-in scorer and shows a notice. The job never fails because of the AI step.

**Local Ollama (free, private)**

1. Install Ollama from https://ollama.com/download.
2. Pull a model:
   ```bat
   ollama pull llama3.2
   ```
3. In ClipForge, turn on **AI clip-picker** and set: Mode **Local**, Base URL `http://localhost:11434`, Model `llama3.2`.
4. Click **Test connection**. This also lists your installed models.

**Ollama Cloud**

1. Create an API key at https://ollama.com/settings/keys.
2. Set: Mode **Cloud**, Base URL `https://ollama.com`, a cloud model name from https://ollama.com/search?c=cloud, and paste your key.
3. Click **Save settings**. The key is sent as `Authorization: Bearer <key>`.

Settings, including the key, are stored in `config.json` next to `app.py`. This file is in `.gitignore`, so it is never committed, and the key is never sent back to the browser.

## GPU transcription

faster-whisper (CTranslate2) needs the CUDA 12 cuBLAS and cuDNN 9 libraries to use an NVIDIA GPU. `setup.bat` installs them automatically from `requirements-gpu.txt` (about 1 GB) when `nvidia-smi` finds a GPU. To install them by hand:

```bat
.venv\Scripts\activate
pip install -r requirements-gpu.txt
```

ClipForge loads these DLLs on start-up, and the top bar then shows "Whisper: NVIDIA GPU". Tested on a GTX 1650 (4 GB). If the libraries can't be loaded, it uses the CPU. Rendering uses NVENC on its own, with no extra libraries.

## Project layout

```
app.py              FastAPI server: UI, jobs API, uploads, downloads, zip, settings
pipeline/           Processing pipeline (no web dependencies, can run on another server)
  runner.py         Orchestrates one job end to end (run_job)
  ingest.py         Filename sanitising, yt-dlp downloads with friendly errors
  audio.py          16 kHz mono WAV extraction and loudness analysis
  transcribe.py     faster-whisper with word timestamps, VAD, GPU/CPU selection
  scoring.py        Sentence units, candidate windows, heuristic score, energy-only fallback
  ai_picker.py      Optional Ollama picker (local or cloud) with validation and fallback
  captions.py       .ass (burn-in, word highlight) and .srt generation
  render.py         ffmpeg rendering (H.264/AAC MP4, faststart, NVENC or x264)
  ffmpeg_utils.py   Locates ffmpeg, checks filters/NVENC, runs ffmpeg, probes media
static/             Single-page UI (HTML/CSS/JS, no build step)
jobs/<job-id>/      One folder per job: source video, clips, .srt/.ass, transcript.json, result.json
models/             Downloaded Whisper models
```

API endpoints: `POST /api/jobs`, `PUT /api/jobs/{id}/upload` (raw streamed body), `GET /api/jobs/{id}/status`, `POST /api/jobs/{id}/cancel`, `GET /api/jobs/{id}/files/{name}`, `GET /api/jobs/{id}/zip`, `DELETE /api/jobs`, `GET/POST /api/config`, `POST /api/ollama/test`, `GET /api/system`.

## Known limits

- **Speed on CPU:** transcription takes roughly 0.1 to 0.3x real time with `base` on a modern CPU, and is much slower with `medium`. A 1-hour video can take 10 to 30 minutes.
- **One job at a time:** extra jobs wait in a queue.
- **Cancel timing:** Cancel takes effect between transcription segments and render steps. During the first download of a Whisper model, the job stops only after the download finishes.
- **Vertical crop:** 9:16 is a fixed centre crop. There is no face or speaker tracking, so off-centre speakers can be cut off.
- **Memory:** the transcription audio is held in memory as 16 kHz float samples, about 230 MB per hour of audio. Everything else streams from disk.
- **Downloads:** some sites block downloads, need a sign-in, or are geo-restricted. Updating yt-dlp (`setup.bat`) fixes many failures. YouTube support uses the bundled Deno runtime that yt-dlp needs.
- **Hindi:** detection and captions work (Nirmala UI font). Hook detection uses a fixed phrase list, so it is less nuanced than for English.
- **Scores** are heuristic signals, not a prediction of views.
- The server listens on `127.0.0.1` only and has no authentication. Don't expose it to a network as-is.
