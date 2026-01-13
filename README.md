# Run this in powershell in the folder that the script is located to install everything necessary for this script to run
#Open this in code (left corner)

# setup_all.ps1
# Eén enkel script (kladblok-stijl): dwingt Python-versie af, maakt .venv, installeert deps + ECGtizer + Poppler,
# zet POPLER_BIN/PATH, en toont zéker de Python-versie der venv.

[CmdletBinding()]
param(
  # Vereischte Python minor-versie (pas dit aan naar uw nood, bv. 3.10 of 3.11)
  [string]$RequiredPython = "3.10",

  # Indien gezet: installeert Python via winget indien 'py -$RequiredPython' ontbreekt
  [switch]$AutoInstallPython,

  # Indien gezet: schrijft POPLER_BIN en PATH ook naar User env (blijvend). Anders alleen in deze sessie.
  [switch]$PersistEnv,

  # Poppler release (Windows build)
  [string]$PopplerVersion = "24.08.0-0"
)

$ErrorActionPreference = "Stop"

# ----------------------------
# 0) Basis paden
# ----------------------------
$Root = $PSScriptRoot
if (-not $Root) { $Root = (Get-Location).Path }

Write-Host "Werkmap (script-dir): $Root"
Write-Host "Vereischte Python:    $RequiredPython"
Write-Host "Poppler versie:       $PopplerVersion"

# ----------------------------
# 1) Helpers
# ----------------------------
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
    throw "Python $ver ontbreekt (of de launcher 'py' ontbreekt). Installeer Python $ver, of draai met -AutoInstallPython."
  }

  if (-not (Get-Command winget -ErrorAction SilentlyContinue)) {
    throw "Python $ver ontbreekt, en winget is niet beschikbaar. Installeer Python $ver handmatig."
  }

  $wingetId = "Python.Python.$ver"
  Write-Host "Python $ver ontbreekt; winget-installe: $wingetId"
  winget install --id $wingetId --silent --accept-package-agreements --accept-source-agreements

  if (-not (Test-PyVersion $ver)) {
    throw "Na winget-install: 'py -$ver' werkt nog niet. Heropen PowerShell en probeer wederom."
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

function Ensure-Directory([string]$p) {
  if (-not (Test-Path $p)) { New-Item -ItemType Directory -Force -Path $p | Out-Null }
}

function Find-PopplerBin([string]$base) {
  if (-not (Test-Path $base)) { return $null }
  $bins = Get-ChildItem -Path $base -Recurse -Directory -ErrorAction SilentlyContinue |
    Where-Object { $_.FullName -match "Library\\bin$" }
  if ($bins -and $bins.Count -ge 1) { return $bins[0].FullName }
  return $null
}

# ----------------------------
# 2) Python verzekeren (juiste versie)
# ----------------------------
Ensure-Python -ver $RequiredPython -autoInstall:$AutoInstallPython
Write-Host "Python OK via launcher: $(& py "-$RequiredPython" -c 'import sys; print(sys.version)')"

# ----------------------------
# 3) venv maken/activeren
# ----------------------------
$VenvDir = Join-Path $Root ".venv"

# Indien venv reeds bestaat, doch met verkeerde Python, herbouw haar
if (Test-Path $VenvDir) {
  $ExistingVenvPy = Join-Path $VenvDir "Scripts\python.exe"
  if (Test-Path $ExistingVenvPy) {
    $majmin = & $ExistingVenvPy -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
    if ($majmin -ne $RequiredPython) {
      Write-Host "[INFO] Bestaande .venv is Python $majmin, doch vereischt is $RequiredPython. Ik herbouw de venv."
      Remove-Item -Recurse -Force $VenvDir
    }
  }
}

if (-not (Test-Path $VenvDir)) {
  Write-Host "Maak venv: $VenvDir"
  & py "-$RequiredPython" -m venv $VenvDir
}

$Activate = Join-Path $VenvDir "Scripts\Activate.ps1"
if (-not (Test-Path $Activate)) { throw "Kan venv-activatie niet vinden: $Activate" }

Write-Host "Activeer venv..."
. $Activate

# Vaste verwijzing naar venv-python (zekerheid)
$VenvPy = Join-Path $VenvDir "Scripts\python.exe"
if (-not (Test-Path $VenvPy)) { throw "Venv-python niet gevonden: $VenvPy" }

# ----------------------------
# 4) pip tools updaten
# ----------------------------
& $VenvPy -m pip install --upgrade pip setuptools wheel

# ----------------------------
# 5) Python packages installeren (pinnen naar uwe werkende set)
# ----------------------------
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
& $VenvPy -m pip install $Pkgs

# Torch CPU (pin)
Write-Host "Installeer torch (CPU) 2.9.1+cpu..."
& $VenvPy -m pip install --index-url "https://download.pytorch.org/whl/cpu" "torch==2.9.1+cpu" -U

# ECGtizer (GitHub zip)
$ECGtizerZip = "https://github.com/UMMISCO/ecgtizer/archive/refs/heads/master.zip"
Write-Host "Installeer ECGtizer van: $ECGtizerZip"
& $VenvPy -m pip install $ECGtizerZip

# ----------------------------
# 6) Poppler (Windows) downloaden & uitpakken naast dit script
# ----------------------------
$PopplerUrl = "https://github.com/oschwartz10612/poppler-windows/releases/download/v$PopplerVersion/Release-$PopplerVersion.zip"
$PopplerZipPath = Join-Path $Root "Release-$PopplerVersion.zip"
$PopplerInstallRoot = Join-Path $Root "poppler\Release-$PopplerVersion"

Ensure-Directory $PopplerInstallRoot

$PopplerBin = Find-PopplerBin $PopplerInstallRoot
if (-not $PopplerBin) {
  Write-Host "Download Poppler: $PopplerUrl"

  try { [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12 } catch {}
  Invoke-WebRequest -Uri $PopplerUrl -OutFile $PopplerZipPath

  Write-Host "Uitpakken naar: $PopplerInstallRoot"
  Expand-Archive -LiteralPath $PopplerZipPath -DestinationPath $PopplerInstallRoot -Force

  $PopplerBin = Find-PopplerBin $PopplerInstallRoot
}

if (-not $PopplerBin) {
  throw "Poppler bin-map niet gevonden na uitpakken onder: $PopplerInstallRoot"
}

Write-Host "Poppler bin gevonden: $PopplerBin"

# ----------------------------
# 7) POPLER_BIN en PATH zetten (sessie; optioneel User)
# ----------------------------
$env:POPLER_BIN = $PopplerBin
if ($env:Path -notlike "*$PopplerBin*") { $env:Path = "$env:Path;$PopplerBin" }

if ($PersistEnv) {
  Write-Host "Schrijf POPLER_BIN en PATH ook naar User environment (blijvend)."
  Set-EnvVarUser "POPLER_BIN" $PopplerBin
  Add-ToUserPathIfMissing $PopplerBin
  Write-Host "Heropen PowerShell opdat User PATH overal geldt."
}

# ----------------------------
# 8) Zelftest + venv Python diagnose (zeker uit de venv)
# ----------------------------
Write-Host ""
Write-Host "=== Zelftest ==="
Write-Host "Test poppler (pdftoppm -h)..."
& (Join-Path $PopplerBin "pdftoppm.exe") -h | Out-Null
Write-Host "Poppler OK."

Write-Host ""
Write-Host "=== Venv Python diagnose (zeker) ==="
& $VenvPy -c "import sys; print('executable:', sys.executable); print('version:   ', sys.version); print('maj.min:   ', f'{sys.version_info.major}.{sys.version_info.minor}')"

Write-Host ""
Write-Host "=== Imports ==="
& $VenvPy -c "import torch; import pdf2image; from ecgtizer.ecgtizer import ECGtizer; print('torch:', torch.__version__); print('pdf2image:', pdf2image.__version__); print('ECGtizer import: OK')"

Write-Host ""
Write-Host "Klaar. Draait nu uw script met:"
Write-Host "  .\.venv\Scripts\Activate.ps1"
Write-Host "  python .\uw_script.py"

