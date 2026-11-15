; Inno Setup script for ClipForge.  Built by build.bat (needs Inno Setup 6: https://jrsoftware.org/isinfo.php).
; Installs per-user (no admin rights) into %LOCALAPPDATA%\Programs\ClipForge.
; User data (clips, Whisper models, settings, GPU pack) lives in %LOCALAPPDATA%\ClipForge.

#define AppName "ClipForge"
#define AppVersion "1.0.0"
#define AppExe "ClipForge.exe"

[Setup]
AppId={{6E1F3C52-8B0D-4C5E-9A7B-2F4D1C9E7A31}
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppName} {#AppVersion}
AppPublisher=ClipForge
AppPublisherURL=https://github.com/Sarthak01819/clipforge
DefaultDirName={localappdata}\Programs\{#AppName}
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
OutputDir=dist
OutputBaseFilename=ClipForge-Setup-{#AppVersion}
SetupIconFile=assets\clipforge.ico
UninstallDisplayIcon={app}\{#AppExe}
Compression=lzma2/ultra64
SolidCompression=yes
LZMAUseSeparateProcess=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0
; The app creates this mutex while running, so Setup can ask the user to close it first.
AppMutex=ClipForgeDesktopApp

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop shortcut"; GroupDescription: "Additional shortcuts:"

[Files]
Source: "dist\ClipForge\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\{#AppName}"; Filename: "{app}\{#AppExe}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExe}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#AppExe}"; Description: "Start {#AppName}"; Flags: nowait postinstall skipifsilent

[Code]
// On uninstall, offer to delete the user's clips, downloaded models, settings and GPU pack.
procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
var
  DataDir: String;
begin
  if CurUninstallStep = usPostUninstall then
  begin
    DataDir := ExpandConstant('{localappdata}\ClipForge');
    if DirExists(DataDir) and not UninstallSilent() then
      if MsgBox('Also delete your ClipForge data (clips, downloaded Whisper models, settings and GPU pack) in:' + #13#10 + DataDir + ' ?',
                mbConfirmation, MB_YESNO or MB_DEFBUTTON2) = IDYES then
        DelTree(DataDir, True, True, True);
  end;
end;
