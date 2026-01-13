# Run this in powershell in the folder that the script is located to install everything necessary for this script to run
# setup_ecg_env.ps1
# Maekt: .venv + installeert packages + ECGtizer + Poppler (Windows) in map naast dit script.
# En dwingt een bepaalde Python-versie af (via "py -3.11" e.d.). Indien ontbrekend: optioneel auto-install via winget.

[CmdletBinding()]
param(
  # Vereischte Python minor-versie (pas dit aan na uw nood)
  [string]$RequiredPython = "3.11",

  # Indien gezet: probeert Python via winget te installeren als 'py -$RequiredPython' ontbreekt
  [switch]$AutoInstallPython,

  # Indien gezet: schrijft POPPLER_BIN en PATH ook naar User environment (blijvend). Anders slechts in deze sessie.
  [switch]$PersistEnv
)

$ErrorActionPreference = "Stop"

# --- 0) Basis paden ---
$Root = $PSScriptRoot
if (-not $Root) { $Root = (Get-Location).Path }

Write-Host "Werkmap (script-dir): $Root"
Write-Host "Vereischte Python: $RequiredPython"

# --- 1) Helpers ---
function Test-PyVersion([string]$ver) {
  if (-not (Get-Command py -ErrorAction SilentlyContinue)) { return $false }
  try {
    & py "-$ver" -c "import sys; print(sys.version)" | Out-Null
    return $true
  } catch { return $false }
}

function Ensure-Python([string]$ver, [switch]$autoInstall) {
  if (Test-PyVersion $ver) { return }

  if (-not $autoInstall) {
    throw "Python $ver ontbreekt (of de launcher 'py' ontbreekt). Installeer Python $ver (met Python Launcher) of draai met -AutoInstallPython."
  }

  if (-not (Get-Command winget -ErrorAction SilentlyContinue)) {
    throw "Python $ver ontbreekt, en winget is niet beschikbaar. Installeer Python $ver handmatig."
  }

  # winget IDs: Python.Python.3.10 / 3.11 / 3.12 ...
  $majmin = $ver
  $wingetId = "Python.Python.$majmin"

  Write-Host "Python $ver ontbreekt; winget-installe: $wingetId"
  winget install --id $wingetId --silent --accept-package-agreements --accept-source-agreements

  # Her-test
  if (-not (Test-PyVersion $ver)) {
    throw "Na winget-install: 'py -$ver' werkt nog niet. Herstart PowerShell en probeer wederom."
  }
}

function Set-EnvVarUser([string]$name, [string]$value) {
  [Environment]::SetEnvironmentVariable($name, $value, "User")
}

function Add-ToUserPathIfMissing([string]$p) {
  $userPath = [Environment]::GetEnvironmentVariable("Path", "User")
  if (-not $userPath) { $userPath = "" }
  if ($userPath -notlike "*$p*") {
    $newUserPath = if ($userPath.Trim().Length -eq 0) { $p } else { "$userPath;$p" }
    [Environment]::SetEnvironmentVariable("Path", $newUserPath, "User")
  }
}

# --- 2) Python verzekeren ---
Ensure-Python -ver $RequiredPython -autoInstall:$AutoInstallPython

Write-Host "Python OK: $(& py "-$RequiredPython" -c 'import sys; print(sys.version)')"

# --- 3) venv maken/activeren ---
$VenvDir = Join-Path $Root ".venv"
if (-not (Test-Path $VenvDir)) {
  Write-Host "Maak venv: $VenvDir"
  & py "-$RequiredPython" -m venv $VenvDir
}

$Activate = Join-Path $VenvDir "Scripts\Activate.ps1"
if (-not (Test-Path $Activate)) { throw "Kan venv-activatie niet vinden: $Activate" }

Write-Host "Activeer venv..."
. $Activate

# --- 4) pip tools updaten ---
python -m pip install --upgrade pip setuptools wheel

# --- 5) Python dependencies installeren (pinnen naar uw bekende werkende set) ---
$Pkgs = @(
  "numpy==1.24.4",
  "pandas==2.0.3",
  "matplotlib==3.7.5",
  "pillow==10.2.0",
  "tqdm==4.67.1",
  "pdf2image==1.17.0",
  "opencv-python==4.9.0.80",
  "scikit-image==0.25.2",
  "scipy==1.15.3",
  "PyYAML==6.0.3",
  "xmltodict==0.13.0",
  "fastdtw==0.3.4"
)

Write-Host "Installeer Python packages..."
python -m pip install $Pkgs

# Torch CPU (pin)
Write-Host "Installeer torch (CPU) 2.9.1+cpu..."
python -m pip install --index-url "https://download.pytorch.org/whl/cpu" "torch==2.9.1+cpu" -U

# --- 6) ECGtizer installeren (GitHub zip; geen git noodig) ---
$ECGtizerZip = "https://github.com/UMMISCO/ecgtizer/archive/refs/heads/master.zip"
Write-Host "Installeer ECGtizer van: $ECGtizerZip"
python -m pip install $ECGtizerZip

# --- 7) Poppler 24.08.0-0 (Windows) downloaden & uitpakken naast dit script ---
# Bron: poppler-windows releases (oschwartz10612)
$PopplerVersion = "24.08.0-0"
$PopplerUrl = "https://github.com/oschwartz10612/poppler-windows/releases/download/v24.08.0-0/Release-24.08.0-0.zip"

$PopplerZipPath = Join-Path $Root "Release-$PopplerVersion.zip"
$PopplerInstallRoot = Join-Path $Root "poppler\Release-$PopplerVersion"
$PopplerBin = Join-Path $PopplerInstallRoot "poppler-24.08.0\Library\bin"

if (-not (Test-Path $PopplerBin)) {
  Write-Host "Download Poppler: $PopplerUrl"

  # TLS fix voor oudere PS/Windows combinaties
  try {
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
  } catch {}

  Invoke-WebRequest -Uri $PopplerUrl -OutFile $PopplerZipPath

  Write-Host "Uitpakken naar: $PopplerInstallRoot"
  New-Item -ItemType Directory -Force -Path $PopplerInstallRoot | Out-Null
  Expand-Archive -LiteralPath $PopplerZipPath -DestinationPath $PopplerInstallRoot -Force
}

if (-not (Test-Path $PopplerBin)) {
  throw "Poppler bin-map niet gevonden na uitpakken: $PopplerBin"
}

# --- 8) POPLER_BIN en PATH zetten (sessie; en optioneel User) ---
Write-Host "Zet POPLER_BIN (sessie) = $PopplerBin"
$env:POPLER_BIN = $PopplerBin

if ($env:Path -notlike "*$PopplerBin*") {
  $env:Path = "$env:Path;$PopplerBin"
}

if ($PersistEnv) {
  Write-Host "Schrijf POPLER_BIN en PATH ook naar User environment (blijvend)."
  Set-EnvVarUser "POPLER_BIN" $PopplerBin
  Add-ToUserPathIfMissing $PopplerBin
  Write-Host "Let wel: heropen PowerShell opdat User PATH overal geldt."
}

# --- 9) Zelftest ---
Write-Host "Test poppler (pdftoppm -h)..."
& (Join-Path $PopplerBin "pdftoppm.exe") -h | Out-Null
Write-Host "Poppler OK."

Write-Host ""
Write-Host "Diagnose:"
python -c "import sys; print('python:', sys.version)"
python -c "import torch; print('torch:', torch.__version__)"
python -c "import pdf2image; print('pdf2image:', pdf2image.__version__)"
python -c "from ecgtizer.ecgtizer import ECGtizer; print('ECGtizer import: OK')"

Write-Host ""
Write-Host "Klaar. Draait nu uw Python-script binnen deze sessie/venv."
Write-Host "Gedenkt: in uw Python-code moet POPLER_BIN bij voorkeur uit de omgeving gelezen worden,"
Write-Host "anders blijft zij wijzen naar C:\Program Files\... in plaats van naar .\poppler\..."
