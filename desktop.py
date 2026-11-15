"""ClipForge desktop app: the same UI in a native window instead of a browser.

Starts the FastAPI server in-process on a free localhost port and shows it with
pywebview (Microsoft Edge WebView2, built into Windows 10/11). User data lives in
%LOCALAPPDATA%\\ClipForge (jobs, Whisper models, config.json, optional GPU pack).

Run from source:  .venv\\Scripts\\python desktop.py
Packaged by build.bat into dist\\ClipForge\\ClipForge.exe and an installer.
"""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path

APP_NAME = "ClipForge"


# --------------------------------------------------------------------------- environment

def _data_dir() -> Path:
    base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
    path = Path(os.environ.get("CLIPFORGE_HOME") or Path(base) / APP_NAME)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _setup_environment() -> Path:
    """Point the server at the user data folder and make bundled tools findable."""
    data = _data_dir()
    os.environ["CLIPFORGE_HOME"] = str(data)
    os.environ["CLIPFORGE_DESKTOP"] = "1"
    # Bundled helper executables (deno.exe, needed by yt-dlp for YouTube) sit next to the app.
    bundle = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    exe_dir = Path(sys.executable).resolve().parent
    os.environ["PATH"] = os.pathsep.join([str(bundle), str(exe_dir), os.environ.get("PATH", "")])

    # A windowed (no console) build has no stdout/stderr: send logs to a file instead.
    if sys.stdout is None or sys.stderr is None:
        log_path = data / "clipforge.log"
        if log_path.exists() and log_path.stat().st_size > 5_000_000:
            log_path.replace(data / "clipforge.old.log")
        stream = open(log_path, "a", encoding="utf-8", buffering=1)  # noqa: SIM115 - lives for the process
        sys.stdout = sys.stdout or stream
        sys.stderr = sys.stderr or stream
    return data


def _message_box(text: str, title: str = APP_NAME) -> None:
    try:
        import ctypes

        ctypes.windll.user32.MessageBoxW(None, text, title, 0x40)
    except Exception:  # noqa: BLE001 - non-Windows or no GUI
        print(text, file=sys.stderr)


def _single_instance() -> bool:
    """Return False if ClipForge is already running (named mutex)."""
    try:
        import ctypes

        # Same name as AppMutex in installer.iss, so Setup can tell when the app is running.
        ctypes.windll.kernel32.CreateMutexW(None, False, "ClipForgeDesktopApp")
        return ctypes.windll.kernel32.GetLastError() != 183  # ERROR_ALREADY_EXISTS
    except Exception:  # noqa: BLE001
        return True


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# --------------------------------------------------------------------------- JS bridge

class Api:
    """Functions the page can call as ``window.pywebview.api.<name>(...)``."""

    def __init__(self, server_module) -> None:
        self._srv = server_module
        self._window = None  # set after the window is created (private: pywebview exposes public attributes to JS)

    def save_file(self, job_id: str, name: str) -> dict:
        """Ask where to save a clip/SRT/zip with a native dialog, then copy it there."""
        import webview

        try:
            src, suggested = self._srv.resolve_download(str(job_id), str(name))
        except Exception as exc:  # noqa: BLE001 - HTTPException carries .detail
            return {"ok": False, "error": getattr(exc, "detail", None) or str(exc)}
        downloads = Path.home() / "Downloads"
        dialog = getattr(getattr(webview, "FileDialog", None), "SAVE", None) or webview.SAVE_DIALOG
        result = self._window.create_file_dialog(dialog, directory=str(downloads if downloads.is_dir() else Path.home()),
                                                save_filename=suggested)
        if not result:
            return {"ok": False, "cancelled": True}
        dest = Path(result[0] if isinstance(result, (list, tuple)) else result)
        try:
            shutil.copyfile(src, dest)
        except OSError as exc:
            return {"ok": False, "error": f"Could not save the file: {exc.strerror or exc}"}
        return {"ok": True, "path": str(dest)}

    def show_in_folder(self, path: str) -> None:
        if path and Path(path).exists():
            subprocess.Popen(["explorer", "/select,", str(Path(path))])  # noqa: S603,S607 - fixed program

    def open_url(self, url: str) -> None:
        """Open external links in the user's normal browser, not inside the app window."""
        if isinstance(url, str) and url.startswith(("https://", "http://")):
            webbrowser.open(url)

    def open_data_folder(self) -> None:
        subprocess.Popen(["explorer", os.environ["CLIPFORGE_HOME"]])  # noqa: S603,S607


# --------------------------------------------------------------------------- main

def main() -> int:
    if not _single_instance():
        _message_box("ClipForge is already running.")
        return 0
    data = _setup_environment()

    import uvicorn
    import webview

    import app as server  # imported after CLIPFORGE_HOME is set, so it uses the data folder

    port = int(os.environ.get("CLIPFORGE_PORT") or 0) or _free_port()  # override only for testing
    config = uvicorn.Config(server.app, host="127.0.0.1", port=port, log_config=None, access_log=False)
    uv = uvicorn.Server(config)
    thread = threading.Thread(target=uv.run, daemon=True, name="server")
    thread.start()
    deadline = time.time() + 30
    while not uv.started and thread.is_alive() and time.time() < deadline:
        time.sleep(0.05)
    if not uv.started:
        _message_box(f"ClipForge could not start its local server.\n\nDetails are in:\n{data / 'clipforge.log'}")
        return 1

    api = Api(server)
    window = webview.create_window(
        APP_NAME, f"http://127.0.0.1:{port}/", js_api=api, width=1240, height=900,
        min_size=(420, 600), background_color="#0f1115", text_select=True,
    )
    api._window = window

    def on_closing() -> None:
        # Stop running jobs (kills ffmpeg) before the process exits.
        for job in list(server.jobs.values()):
            job.cancel_event.set()
        server.gpu_pack.cancel_install()

    window.events.closing += on_closing
    try:
        webview.start(gui="edgechromium", private_mode=False, storage_path=str(data / "webview"))
    except Exception as exc:  # noqa: BLE001 - usually a missing WebView2 runtime
        _message_box(
            "ClipForge needs the Microsoft Edge WebView2 Runtime, which is part of Windows 11 "
            "and most Windows 10 PCs.\n\nInstall it from https://go.microsoft.com/fwlink/p/?LinkId=2124703 "
            f"and start ClipForge again.\n\n({exc.__class__.__name__}: {exc})"
        )
        return 1
    finally:
        uv.should_exit = True
    # Whisper/ffmpeg worker threads may still be unwinding; don't wait for them.
    os._exit(0)


if __name__ == "__main__":
    sys.exit(main())
