@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\activate.bat" (
  echo ClipForge is not set up yet. Running setup.bat first ...
  call setup.bat
  if not exist ".venv\Scripts\activate.bat" exit /b 1
)
call ".venv\Scripts\activate.bat"
echo Starting ClipForge on http://localhost:7860  (press Ctrl+C to stop)
rem Open the browser a few seconds later, once the server is listening.
start "" /min powershell -NoProfile -Command "Start-Sleep -Seconds 4; Start-Process 'http://localhost:7860'"
python app.py
pause
