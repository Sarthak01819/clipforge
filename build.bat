@echo off
rem Builds the ClipForge desktop app:
rem   dist\ClipForge\ClipForge.exe        (portable folder)
rem   dist\ClipForge-Setup-1.0.0.exe      (compressed installer, if Inno Setup 6 is installed)
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Run setup.bat first.
  exit /b 1
)
echo Installing build tools ...
".venv\Scripts\python.exe" -m pip install -q -r requirements-build.txt || exit /b 1

echo Building the app with PyInstaller ...
".venv\Scripts\pyinstaller.exe" --noconfirm --clean clipforge.spec || exit /b 1

set "ISCC="
if exist "%LOCALAPPDATA%\Programs\Inno Setup 6\ISCC.exe" set "ISCC=%LOCALAPPDATA%\Programs\Inno Setup 6\ISCC.exe"
if exist "%ProgramFiles(x86)%\Inno Setup 6\ISCC.exe" set "ISCC=%ProgramFiles(x86)%\Inno Setup 6\ISCC.exe"
if not defined ISCC (
  echo Inno Setup 6 not found - skipping the installer. Get it with:  winget install JRSoftware.InnoSetup
  echo The app is ready in dist\ClipForge\ClipForge.exe
  exit /b 0
)
echo Building the installer ...
"%ISCC%" /Q installer.iss || exit /b 1
echo Done: dist\ClipForge-Setup-1.0.0.exe
