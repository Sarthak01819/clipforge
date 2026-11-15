@echo off
setlocal
cd /d "%~dp0"
echo === ClipForge setup ===

set "PY="
where py >nul 2>nul && set "PY=py -3"
if not defined PY (
  where python >nul 2>nul && set "PY=python"
)
if not defined PY (
  echo [ERROR] Python 3.10+ was not found. Install it from https://www.python.org/downloads/ and tick "Add to PATH".
  pause
  exit /b 1
)

%PY% -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)"
if errorlevel 1 (
  echo [ERROR] Python 3.10 or newer is required.
  %PY% --version
  pause
  exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
  echo Creating virtual environment in .venv ...
  %PY% -m venv .venv
  if errorlevel 1 (
    echo [ERROR] Could not create the virtual environment.
    pause
    exit /b 1
  )
)

echo Installing / updating packages (this can take a few minutes) ...
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install --upgrade -r requirements.txt
if errorlevel 1 (
  echo [ERROR] Package installation failed. Check your internet connection and try again.
  pause
  exit /b 1
)

where nvidia-smi >nul 2>nul
if errorlevel 1 (
  echo No NVIDIA GPU detected - Whisper will run on the CPU.
) else (
  echo NVIDIA GPU detected - installing CUDA libraries for GPU transcription ^(about 1 GB^) ...
  ".venv\Scripts\python.exe" -m pip install --upgrade -r requirements-gpu.txt
  if errorlevel 1 echo [WARNING] GPU libraries failed to install - Whisper will use the CPU.
)

where ffmpeg >nul 2>nul
if errorlevel 1 (
  echo FFmpeg not on PATH - ClipForge will use the copy bundled with imageio-ffmpeg.
) else (
  echo Using system FFmpeg from PATH.
)

echo.
echo Setup complete. Start ClipForge with run.bat
pause
