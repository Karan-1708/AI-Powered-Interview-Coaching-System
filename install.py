"""
AI Interview Coach — Universal Setup Engine
============================================
Handles system scanning, Python validation, virtual environment management,
FFmpeg portable install, GPU-aware PyTorch selection, and .env creation.

Designed to be friendly for non-technical users.
Run directly:  python install.py
Or called by:  start.bat / start.sh
"""

import os
import sys
import platform
import subprocess
import shutil
import json
import struct
import urllib.request
import zipfile
import tarfile
import tempfile
import time
from pathlib import Path

# ══════════════════════════════════════════════════════════════════════════════
#  ANSI COLOUR HELPERS  (enabled on Windows 10+, Mac, Linux)
# ══════════════════════════════════════════════════════════════════════════════

def _enable_ansi_windows():
    if platform.system() == "Windows":
        try:
            import ctypes
            k = ctypes.windll.kernel32
            k.SetConsoleMode(k.GetStdHandle(-11), 7)
        except Exception:
            pass

_enable_ansi_windows()

class C:
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"
    RED     = "\033[91m"
    GREEN   = "\033[92m"
    YELLOW  = "\033[93m"
    BLUE    = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN    = "\033[96m"
    WHITE   = "\033[97m"

# ── Print helpers ─────────────────────────────────────────────────────────────

def banner():
    print(f"""
{C.CYAN}{C.BOLD}╔══════════════════════════════════════════════════════════╗
║          AI Interview Coach  ·  Setup Engine             ║
║          Intelligent Installer  ·  All Platforms         ║
╚══════════════════════════════════════════════════════════╝{C.RESET}
""")

def section(n, total, title):
    bar = "─" * 52
    print(f"\n{C.BOLD}{C.BLUE}┌{bar}┐{C.RESET}")
    print(f"{C.BOLD}{C.BLUE}│{C.RESET}  {C.BOLD}Step {n}/{total}{C.RESET}  {C.WHITE}{title:<44}{C.RESET}{C.BOLD}{C.BLUE}│{C.RESET}")
    print(f"{C.BOLD}{C.BLUE}└{bar}┘{C.RESET}")

def ok(msg):    print(f"  {C.GREEN}✔{C.RESET}  {msg}")
def info(msg):  print(f"  {C.CYAN}ℹ{C.RESET}  {msg}")
def warn(msg):  print(f"  {C.YELLOW}⚠{C.RESET}  {msg}")
def err(msg):   print(f"  {C.RED}✘{C.RESET}  {C.RED}{msg}{C.RESET}")

def fatal(msg):
    print(f"\n{C.RED}{C.BOLD}  ✘  FATAL: {msg}{C.RESET}")
    print(f"{C.RED}  Setup cannot continue. Fix the issue above, then try again.{C.RESET}\n")
    sys.exit(1)

def progress_bar(label, pct, width=28):
    filled = int(width * pct / 100)
    bar    = f"{C.CYAN}{'█' * filled}{'░' * (width - filled)}{C.RESET}"
    print(f"\r  {C.DIM}{label}{C.RESET} [{bar}] {C.BOLD}{pct:3d}%{C.RESET}", end="", flush=True)

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 1 — SYSTEM SCAN
# ══════════════════════════════════════════════════════════════════════════════

def scan_system() -> dict:
    """Collect OS, CPU, RAM, GPU details. Returns a structured dict."""
    s = {}

    # OS
    s["os"]         = platform.system()           # Windows / Darwin / Linux
    s["os_release"] = platform.release()
    s["os_version"] = platform.version()
    s["machine"]    = platform.machine()          # x86_64 / arm64 / AMD64
    s["arch"]       = platform.architecture()[0]  # 32bit / 64bit

    # CPU
    s["cpu_name"]  = _cpu_name()
    s["cpu_cores"] = os.cpu_count() or 1

    # RAM
    s["ram_gb"] = _ram_gb()

    # GPU
    s["gpu"] = _detect_gpu()

    # Python
    s["python_version"] = platform.python_version()
    s["python_tuple"]   = tuple(sys.version_info[:3])
    s["python_exe"]     = sys.executable

    return s


def _cpu_name() -> str:
    try:
        if platform.system() == "Windows":
            r = subprocess.run("wmic cpu get Name", shell=True, capture_output=True, text=True)
            lines = [l.strip() for l in r.stdout.splitlines() if l.strip() and l.strip() != "Name"]
            return lines[0] if lines else "Unknown CPU"
        elif platform.system() == "Darwin":
            r = subprocess.run("sysctl -n machdep.cpu.brand_string", shell=True, capture_output=True, text=True)
            return r.stdout.strip() or "Apple Silicon"
        else:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if "model name" in line:
                        return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return "Unknown CPU"


def _ram_gb() -> float:
    try:
        if platform.system() == "Windows":
            r = subprocess.run("wmic OS get TotalVisibleMemorySize", shell=True, capture_output=True, text=True)
            lines = [l.strip() for l in r.stdout.splitlines() if l.strip() and not l.strip().startswith("Total")]
            return round(int(lines[0]) / 1024 / 1024, 1) if lines else 0.0
        elif platform.system() == "Darwin":
            r = subprocess.run("sysctl -n hw.memsize", shell=True, capture_output=True, text=True)
            return round(int(r.stdout.strip()) / 1024 ** 3, 1)
        else:
            with open("/proc/meminfo") as f:
                for line in f:
                    if "MemTotal" in line:
                        return round(int(line.split()[1]) / 1024 / 1024, 1)
    except Exception:
        pass
    return 0.0


def _detect_gpu() -> dict:
    """Returns {vendor, name, cuda, mps, backend}"""
    gpu = {"vendor": "none", "name": "None detected", "cuda": False, "mps": False, "backend": "cpu"}

    # NVIDIA via nvidia-smi
    if shutil.which("nvidia-smi"):
        try:
            r = subprocess.run(
                "nvidia-smi --query-gpu=name --format=csv,noheader",
                shell=True, capture_output=True, text=True
            )
            if r.returncode == 0 and r.stdout.strip():
                gpu.update(vendor="nvidia", name=r.stdout.splitlines()[0].strip(),
                           cuda=True, backend="cuda")
                return gpu
        except Exception:
            pass

    # Apple Silicon MPS
    if platform.system() == "Darwin" and platform.machine() in ("arm64", "arm64e"):
        gpu.update(vendor="apple", name="Apple Silicon GPU (MPS)", mps=True, backend="mps")
        return gpu

    # Windows fallback — show AMD/Intel GPU name even if no CUDA
    if platform.system() == "Windows":
        try:
            r = subprocess.run("wmic path win32_VideoController get name", shell=True, capture_output=True, text=True)
            lines = [l.strip() for l in r.stdout.splitlines() if l.strip() and l.strip() != "Name"]
            if lines:
                name = lines[0]
                vendor = "amd" if ("amd" in name.lower() or "radeon" in name.lower()) else "intel" if "intel" in name.lower() else "other"
                gpu.update(vendor=vendor, name=name, backend="cpu")
                return gpu
        except Exception:
            pass

    return gpu


def print_system_report(s: dict):
    gpu_label = s["gpu"]["name"]
    accel = ""
    if s["gpu"]["cuda"]:
        accel = f"  {C.GREEN}[CUDA]{C.RESET}"
    elif s["gpu"]["mps"]:
        accel = f"  {C.GREEN}[MPS]{C.RESET}"

    print(f"""
  {C.DIM}┌───────────────────────────────────────────────┐{C.RESET}
  {C.DIM}│{C.RESET}  {C.BOLD}System Scan Results{C.RESET}
  {C.DIM}│{C.RESET}
  {C.DIM}│{C.RESET}  {C.DIM}Operating System{C.RESET}   {C.WHITE}{s['os']} {s['os_release']}{C.RESET}
  {C.DIM}│{C.RESET}  {C.DIM}Architecture     {C.RESET}   {C.WHITE}{s['machine']}  ({s['arch']}){C.RESET}
  {C.DIM}│{C.RESET}  {C.DIM}CPU              {C.RESET}   {C.WHITE}{s['cpu_name']}{C.RESET}
  {C.DIM}│{C.RESET}  {C.DIM}CPU Cores        {C.RESET}   {C.WHITE}{s['cpu_cores']}{C.RESET}
  {C.DIM}│{C.RESET}  {C.DIM}RAM              {C.RESET}   {C.WHITE}{s['ram_gb']} GB{C.RESET}
  {C.DIM}│{C.RESET}  {C.DIM}GPU              {C.RESET}   {C.WHITE}{gpu_label}{C.RESET}{accel}
  {C.DIM}│{C.RESET}  {C.DIM}Python           {C.RESET}   {C.WHITE}{s['python_version']}{C.RESET}
  {C.DIM}│{C.RESET}
  {C.DIM}│{C.RESET}  {C.DIM}PyTorch Backend  {C.RESET}   {C.CYAN}{C.BOLD}{s['gpu']['backend'].upper()}{C.RESET}
  {C.DIM}└───────────────────────────────────────────────┘{C.RESET}""")

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 2 — PYTHON VERSION GATE
# ══════════════════════════════════════════════════════════════════════════════

MIN_PYTHON = (3, 11)

def check_python():
    major, minor, micro = sys.version_info[:3]
    ver = f"{major}.{minor}.{micro}"
    if (major, minor) < MIN_PYTHON:
        fatal(
            f"Python {ver} is too old.\n"
            f"  This application requires Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]} or newer.\n"
            f"  Download the latest version from:  https://www.python.org/downloads/"
        )
    ok(f"Python {ver}  ✓")

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 3 — VIRTUAL ENVIRONMENT  (interactive scan + selection)
# ══════════════════════════════════════════════════════════════════════════════

# Module-level — updated by ensure_venv() so all helpers stay in sync
VENV_DIR = Path(".venv")

def _venv_python() -> Path:
    if platform.system() == "Windows":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"

def _venv_pip() -> Path:
    if platform.system() == "Windows":
        return VENV_DIR / "Scripts" / "pip.exe"
    return VENV_DIR / "bin" / "pip"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _probe_venv(venv_path: Path) -> dict | None:
    """
    Probe a candidate venv directory.
    Returns {path, python_exe, version, healthy} or None if it isn't a venv.
    """
    if platform.system() == "Windows":
        py = venv_path / "Scripts" / "python.exe"
    else:
        py = venv_path / "bin" / "python"

    if not py.exists():
        return None

    r = subprocess.run([str(py), "--version"], capture_output=True, text=True)
    if r.returncode != 0:
        return {"path": venv_path, "python_exe": py, "version": "?", "healthy": False}

    ver = r.stdout.strip() or r.stderr.strip()   # "Python 3.11.9"
    ver = ver.replace("Python ", "").strip()
    return {"path": venv_path, "python_exe": py, "version": ver, "healthy": True}


def _scan_for_venvs(root: Path = Path("."), max_depth: int = 2) -> list[dict]:
    """
    Walk up to max_depth directory levels and identify venv directories by
    STRUCTURE, not by name — so custom names like 'ai-venv', 'my-env',
    'project-3.11', etc. are all detected correctly.

    A directory is treated as a venv if it contains the Python binary at the
    expected platform path:
      Windows : <dir>/Scripts/python.exe
      Mac/Lin : <dir>/bin/python  OR  <dir>/bin/python3

    Also skips common non-venv dirs to keep the scan fast.
    """
    candidates  = []
    IS_WINDOWS  = platform.system() == "Windows"

    # Folders that are definitely not venvs — skip entirely for speed
    SKIP_NAMES = {
        "node_modules", "__pycache__", ".git", ".hg", ".svn",
        "dist", "build", "site-packages", ".mypy_cache", ".pytest_cache",
        ".tox", ".nox", "htmlcov", ".eggs",
    }

    def _looks_like_venv(path: Path) -> bool:
        """True if the directory contains a Python binary at the venv-standard location."""
        if IS_WINDOWS:
            return (path / "Scripts" / "python.exe").exists()
        return (path / "bin" / "python").exists() or (path / "bin" / "python3").exists()

    def _walk(path: Path, depth: int):
        if depth > max_depth:
            return
        try:
            for child in sorted(path.iterdir()):
                if not child.is_dir():
                    continue
                if child.name in SKIP_NAMES:
                    continue
                if _looks_like_venv(child):
                    probe = _probe_venv(child)
                    if probe:
                        candidates.append(probe)
                    # Don't recurse into a venv itself
                else:
                    _walk(child, depth + 1)
        except PermissionError:
            pass

    _walk(root, 0)

    # Healthy venvs first, then broken ones; alphabetical within each group
    candidates.sort(key=lambda v: (0 if v["healthy"] else 1, str(v["path"])))
    return candidates


def _ask_choice(prompt: str, options: list[str], default: int = 0) -> int:
    """
    Print a numbered menu and return the zero-based index of the chosen option.
    Works with plain stdin — no external libraries needed.
    """
    print()
    for i, opt in enumerate(options):
        marker = f"{C.CYAN}►{C.RESET}" if i == default else " "
        print(f"  {marker} {C.BOLD}[{i + 1}]{C.RESET}  {opt}")
    print()

    while True:
        try:
            raw = input(f"  {C.YELLOW}Enter number (default {default + 1}): {C.RESET}").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            fatal("Setup cancelled by user.")

        if raw == "":
            return default
        if raw.isdigit():
            idx = int(raw) - 1
            if 0 <= idx < len(options):
                return idx
        print(f"  {C.RED}Invalid choice — please enter a number between 1 and {len(options)}.{C.RESET}")


def _create_new_venv(target: Path):
    """Create a fresh venv at target, fatal on failure."""
    if target.exists():
        warn(f"{target} already exists — removing it first...")
        shutil.rmtree(target)
    info(f"Creating virtual environment at {C.CYAN}{target}{C.RESET} …")
    r = subprocess.run([sys.executable, "-m", "venv", str(target)], capture_output=True, text=True)
    if r.returncode != 0:
        fatal(f"Could not create virtual environment:\n  {r.stderr}")
    ok(f"Virtual environment created at {target}")


# ── Main entry point for Step 3 ───────────────────────────────────────────────

def ensure_venv():
    """
    Scan for existing venvs, present options to the user, and set the
    module-level VENV_DIR so all downstream helpers (_venv_python, etc.) work.
    """
    global VENV_DIR

    info("Scanning for existing virtual environments…")
    found = _scan_for_venvs()

    # ── Build the menu ────────────────────────────────────────────────────────
    options   = []
    meta      = []   # parallel list of (kind, data) tuples

    for v in found:
        status = f"{C.GREEN}healthy{C.RESET}" if v["healthy"] else f"{C.RED}damaged{C.RESET}"
        label  = (
            f"Use existing  {C.CYAN}{v['path']}{C.RESET}"
            f"  —  Python {v['version']}  [{status}]"
        )
        options.append(label)
        meta.append(("existing", v))

    options.append(f"Create a {C.BOLD}new{C.RESET} .venv in this folder  {C.DIM}(recommended){C.RESET}")
    meta.append(("new", None))

    # ── Prompt ───────────────────────────────────────────────────────────────
    if found:
        print(f"\n  {C.WHITE}Found {len(found)} virtual environment(s). Choose one to use:{C.RESET}")
        default_idx = 0   # first found (healthy) is pre-selected
    else:
        info("No existing virtual environments found.")
        print(f"  {C.WHITE}Choose an option:{C.RESET}")
        default_idx = len(options) - 1   # only option is "create new"

    chosen_idx  = _ask_choice("Select virtual environment", options, default=default_idx)
    kind, data  = meta[chosen_idx]

    # ── Act on choice ─────────────────────────────────────────────────────────
    if kind == "new":
        target   = Path(".venv")
        _create_new_venv(target)
        VENV_DIR = target

    else:   # "existing"
        venv_path = data["path"]
        VENV_DIR  = venv_path

        if not data["healthy"]:
            warn(f"The selected environment at {venv_path} appears damaged.")
            print(f"  {C.WHITE}What would you like to do?{C.RESET}")
            fix_choice = _ask_choice(
                "Fix damaged venv",
                [
                    f"Rebuild it  {C.DIM}(delete and recreate at the same path){C.RESET}",
                    f"Use it anyway  {C.DIM}(may cause errors){C.RESET}",
                ],
                default=0,
            )
            if fix_choice == 0:
                _create_new_venv(venv_path)
            else:
                warn("Proceeding with damaged environment — some packages may not load.")
        else:
            ok(f"Using  {C.CYAN}{venv_path}{C.RESET}  (Python {data['version']})")

    print(f"  {C.DIM}Active venv → {VENV_DIR.resolve()}{C.RESET}")

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 4 — FFMPEG  (portable, installed into ./bin/)
# ══════════════════════════════════════════════════════════════════════════════

BIN_DIR = Path("bin")

# Direct download URLs for portable FFmpeg
FFMPEG_URLS = {
    "Windows": "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip",
    "Linux":   "https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz",
    # macOS: installed via Homebrew (no good universal portable binary)
}

def ensure_ffmpeg():
    os_name = platform.system()

    # 1. Already on system PATH?
    if shutil.which("ffmpeg"):
        r = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True)
        ver = r.stdout.splitlines()[0] if r.stdout else "found"
        ok(f"FFmpeg on PATH  ({ver[:60]})")
        return

    # 2. Already in ./bin/?
    local = _local_ffmpeg_bin()
    if local:
        ok(f"Portable FFmpeg at {local}")
        _register_bin_dir()
        return

    # 3. Try OS package manager (Mac only — no reliable portable binary)
    if os_name == "Darwin":
        if shutil.which("brew"):
            info("Installing FFmpeg via Homebrew…")
            r = subprocess.run("brew install ffmpeg", shell=True)
            if r.returncode == 0:
                ok("FFmpeg installed via Homebrew")
                return
        warn("Homebrew not found. Install FFmpeg manually: https://ffmpeg.org/download.html")
        return

    if os_name == "Linux" and not FFMPEG_URLS.get("Linux"):
        _try_apt_ffmpeg()
        return

    # 4. Download portable binary
    url = FFMPEG_URLS.get(os_name)
    if not url:
        warn(f"Automatic FFmpeg install not supported on {os_name}. Please install it manually.")
        return

    _download_portable_ffmpeg(url, os_name)


def _local_ffmpeg_bin() -> Path | None:
    name = "ffmpeg.exe" if platform.system() == "Windows" else "ffmpeg"
    # Check ./bin/ first, then project root
    for search in [BIN_DIR, Path(".")]:
        candidate = search / name
        if candidate.exists():
            return candidate
    return None


def _register_bin_dir():
    """Add ./bin to PATH so the app can find ffmpeg at runtime."""
    bin_abs = str(BIN_DIR.resolve())
    os.environ["PATH"] = bin_abs + os.pathsep + os.environ.get("PATH", "")


def _try_apt_ffmpeg():
    info("Attempting apt install of FFmpeg…")
    r = subprocess.run("sudo apt-get update -qq && sudo apt-get install -y ffmpeg", shell=True)
    if r.returncode == 0:
        ok("FFmpeg installed via apt")
    else:
        warn("Could not install FFmpeg automatically. Please run:  sudo apt install ffmpeg")


def _download_portable_ffmpeg(url: str, os_name: str):
    BIN_DIR.mkdir(exist_ok=True)
    suffix  = ".zip" if url.endswith(".zip") else ".tar.xz"
    tmp     = Path(tempfile.mktemp(suffix=suffix))

    info(f"Downloading portable FFmpeg…")
    try:
        urllib.request.urlretrieve(url, tmp, _make_dl_hook("FFmpeg"))
        print()  # newline after progress bar
    except Exception as e:
        warn(f"Download failed: {e}")
        warn("Please install FFmpeg manually from https://ffmpeg.org/download.html")
        tmp.unlink(missing_ok=True)
        return

    info("Extracting…")
    try:
        if suffix == ".zip":
            with zipfile.ZipFile(tmp) as z:
                for member in z.namelist():
                    base = os.path.basename(member)
                    if base in ("ffmpeg.exe", "ffprobe.exe"):
                        dest = BIN_DIR / base
                        with z.open(member) as src, open(dest, "wb") as dst:
                            shutil.copyfileobj(src, dst)
        else:  # tar.xz (Linux static build)
            with tarfile.open(tmp) as t:
                for member in t.getmembers():
                    if member.name.endswith("/ffmpeg") or member.name.endswith("/ffprobe"):
                        member.name = os.path.basename(member.name)
                        t.extract(member, BIN_DIR)
    except Exception as e:
        warn(f"Extraction failed: {e}")
        tmp.unlink(missing_ok=True)
        return

    tmp.unlink(missing_ok=True)

    local = _local_ffmpeg_bin()
    if local:
        if os_name != "Windows":
            local.chmod(0o755)
            ffprobe = BIN_DIR / "ffprobe"
            if ffprobe.exists():
                ffprobe.chmod(0o755)
        _register_bin_dir()
        ok(f"Portable FFmpeg ready at {local}")
    else:
        warn("FFmpeg binary not found after extraction — please install manually.")


def _make_dl_hook(label: str):
    def hook(block_num, block_size, total_size):
        if total_size > 0:
            pct = min(100, block_num * block_size * 100 // total_size)
            progress_bar(label, pct)
    return hook

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 5 — DEPENDENCIES  (PyTorch + app packages)
# ══════════════════════════════════════════════════════════════════════════════

APP_PACKAGES = [
    "streamlit",
    "requests",
    "pandas",
    "plotly",
    "fpdf2",
    "Pillow",
    "markdown",
    "psutil",
    "py-cpuinfo",
    "pypdf",
    "edge-tts",
    "python-dotenv",
    "faster-whisper",
    "librosa",
    "soundfile",
    "numpy",
    "uvicorn[standard]",
    "fastapi",
    "python-multipart",
    "httpx",
]


# PyTorch wheel index URLs
TORCH_INDEXES = {
    "cuda_new":  "https://download.pytorch.org/whl/cu124",   # Python 3.11–3.12, CUDA 12.4
    "cuda_old":  "https://download.pytorch.org/whl/cu121",   # Python 3.11, CUDA 12.1
    "cpu":       "https://download.pytorch.org/whl/cpu",
    "default":   None,   # PyPI default (Mac MPS, or fallback)
}


def install_dependencies(sys_info: dict):
    python = str(_venv_python())

    # ── Upgrade pip ──────────────────────────────────────────────────────────
    info("Upgrading pip…")
    subprocess.run(
        [python, "-m", "pip", "install", "--upgrade", "pip", "--quiet"],
        check=False
    )

    # ── Select and install PyTorch ───────────────────────────────────────────
    _install_torch(python, sys_info)

    # ── Install app packages ─────────────────────────────────────────────────
    info("Installing application packages…")
    _pip_install(python, APP_PACKAGES)
    ok("All packages installed")


def _get_installed_torch_info(python: str) -> dict | None:
    """
    Returns dict with keys {version, is_cpu_only, cuda_version} if torch is
    installed inside the venv, or None if torch is not installed at all.

    'is_cpu_only' is True when the wheel was built without CUDA support
    (i.e. torch.version.cuda is None).
    """
    probe = (
        "import torch, json; "
        "print(json.dumps({"
        "'version': torch.__version__, "
        "'cuda_version': torch.version.cuda, "
        "'is_cpu_only': torch.version.cuda is None"
        "}))"
    )
    r = subprocess.run([python, "-c", probe], capture_output=True, text=True)
    if r.returncode != 0:
        return None   # torch not installed
    try:
        return json.loads(r.stdout.strip())
    except Exception:
        return None


def _force_uninstall_torch(python: str):
    """Aggressively remove all torch-related packages from the venv."""
    packages = ["torch", "torchaudio", "torchvision", "torchtext", "triton"]
    subprocess.run(
        [python, "-m", "pip", "uninstall"] + packages + ["-y"],
        capture_output=True, text=True
    )
    # Run twice — some versions leave stubs behind
    subprocess.run(
        [python, "-m", "pip", "uninstall"] + packages + ["-y"],
        capture_output=True, text=True
    )


def _install_torch(python: str, sys_info: dict):
    backend  = sys_info["gpu"]["backend"]
    py_tuple = sys_info["python_tuple"]

    # ── Determine the correct CUDA index for this machine ────────────────────
    # Python 3.12+ → CUDA 12.4   |   Python 3.11 and older → CUDA 12.1
    cuda_index = TORCH_INDEXES["cuda_new"] if py_tuple >= (3, 12) else TORCH_INDEXES["cuda_old"]
    cuda_label = "CUDA 12.4" if py_tuple >= (3, 12) else "CUDA 12.1"

    # ── Check what is already installed ──────────────────────────────────────
    existing = _get_installed_torch_info(python)

    if existing:
        installed_ver = existing["version"]

        if backend == "cuda" and existing["is_cpu_only"]:
            # ── BAD: NVIDIA GPU present but a CPU-only torch wheel is installed ──
            warn(
                f"PyTorch {installed_ver} is installed but it is a CPU-only build.\n"
                f"  Your NVIDIA GPU requires a CUDA build — replacing it now…"
            )
            _force_uninstall_torch(python)
            info(f"Force-uninstalled CPU torch. Installing {cuda_label} build…")
            _pip_install_indexed(python, ["torch", "torchaudio"], cuda_index, cuda_label)
            return

        if backend == "cuda" and not existing["is_cpu_only"]:
            # CUDA torch already installed — check it matches the right cu version
            installed_cuda = existing.get("cuda_version") or ""
            expected_cu    = "12.4" if py_tuple >= (3, 12) else "12.1"
            if not installed_cuda.startswith(expected_cu.replace(".", "")):
                warn(
                    f"PyTorch {installed_ver} is CUDA-enabled but uses CUDA {installed_cuda}.\n"
                    f"  Expected CUDA {expected_cu} for your Python version — reinstalling…"
                )
                _force_uninstall_torch(python)
                _pip_install_indexed(python, ["torch", "torchaudio"], cuda_index, cuda_label)
            else:
                ok(f"PyTorch {installed_ver} ({cuda_label}) already installed — skipping")
            return

        if backend == "mps":
            ok(f"PyTorch {installed_ver} already installed (MPS) — skipping")
            return

        # CPU backend — any existing torch is fine
        ok(f"PyTorch {installed_ver} already installed (CPU) — skipping")
        return

    # ── torch is NOT installed at all — fresh install ────────────────────────
    if backend == "cuda":
        info(f"NVIDIA GPU detected — installing PyTorch ({cuda_label})…")
        _pip_install_indexed(python, ["torch", "torchaudio"], cuda_index, cuda_label)

    elif backend == "mps":
        info("Apple Silicon detected — installing PyTorch with MPS support…")
        _pip_install(python, ["torch", "torchaudio"])
        ok("PyTorch (MPS / Apple Silicon) installed")

    else:
        info("No GPU detected — installing PyTorch (CPU mode)…")
        _pip_install_indexed(python, ["torch", "torchaudio"], TORCH_INDEXES["cpu"], "CPU")


def _pip_install(python: str, packages: list):
    cmd = [python, "-m", "pip", "install"] + packages + ["--quiet"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        warn(f"Some packages may have failed:\n  {r.stderr[:300]}")


def _pip_install_indexed(python: str, packages: list, index_url: str, label: str):
    info(f"Installing PyTorch ({label}) — this can take a few minutes…")
    cmd = [python, "-m", "pip", "install"] + packages + [
        "--index-url", index_url,
        "--force-reinstall",
        "--no-cache-dir",
        "--quiet",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        warn(f"PyTorch ({label}) install issue: {r.stderr[:200]}")
        warn("Falling back to CPU-only PyTorch…")
        _pip_install_indexed(python, packages, TORCH_INDEXES["cpu"], "CPU fallback")
    else:
        ok(f"PyTorch ({label}) installed")

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 6 — .env FILE
# ══════════════════════════════════════════════════════════════════════════════

def ensure_env_file():
    env_path     = Path(".env")
    example_path = Path(".env.example")

    if env_path.exists():
        ok(".env file already exists")
        return

    if example_path.exists():
        shutil.copy(example_path, env_path)
        ok(".env created from .env.example")
        info("Open .env to fill in your API keys before first use.")
    else:
        warn(".env.example not found — creating a blank .env")
        env_path.write_text(
            "# AI Interview Coach — Environment Variables\n"
            "# Add your API keys and settings here.\n\n"
            "ANTHROPIC_API_KEY=\n"
        )

# ══════════════════════════════════════════════════════════════════════════════
#  VERIFY INSTALL
# ══════════════════════════════════════════════════════════════════════════════

def verify_install(sys_info: dict):
    python  = str(_venv_python())
    backend = sys_info["gpu"]["backend"]

    checks = {
        "streamlit":      "import streamlit",
        "fastapi":        "import fastapi",
        "torch":          "import torch",
        "faster_whisper": "import faster_whisper",
    }

    all_ok = True
    for name, stmt in checks.items():
        r = subprocess.run([python, "-c", stmt], capture_output=True)
        if r.returncode == 0:
            ok(f"{name}")
        else:
            err(f"{name}  ← could not import")
            all_ok = False

    # Check torch backend
    if backend == "cuda":
        r = subprocess.run([python, "-c", "import torch; assert torch.cuda.is_available()"], capture_output=True)
        if r.returncode == 0:
            ok("CUDA acceleration verified")
        else:
            warn("CUDA not detected at runtime — will fall back to CPU")
    elif backend == "mps":
        r = subprocess.run([python, "-c", "import torch; assert torch.backends.mps.is_available()"], capture_output=True)
        if r.returncode == 0:
            ok("MPS (Apple Silicon) acceleration verified")
        else:
            warn("MPS not available at runtime — will fall back to CPU")

    return all_ok

# ══════════════════════════════════════════════════════════════════════════════
#  MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

TOTAL_STEPS = 6

def setup():
    banner()

    # ── Step 1: System Scan ──────────────────────────────────────────────────
    section(1, TOTAL_STEPS, "Scanning Your System")
    sys_info = scan_system()
    print_system_report(sys_info)

    # ── Step 2: Python Version ───────────────────────────────────────────────
    section(2, TOTAL_STEPS, "Checking Python Version")
    check_python()

    # ── Step 3: Virtual Environment ──────────────────────────────────────────
    section(3, TOTAL_STEPS, "Preparing Virtual Environment")
    ensure_venv()

    # ── Step 4: FFmpeg ───────────────────────────────────────────────────────
    section(4, TOTAL_STEPS, "Checking FFmpeg")
    ensure_ffmpeg()

    # ── Step 5: Dependencies ─────────────────────────────────────────────────
    section(5, TOTAL_STEPS, "Installing Dependencies")
    install_dependencies(sys_info)

    # ── Step 6: Environment File ─────────────────────────────────────────────
    section(6, TOTAL_STEPS, "Setting Up Configuration")
    ensure_env_file()

    # ── Final Verification ───────────────────────────────────────────────────
    print(f"\n{C.BOLD}{C.MAGENTA}  Verifying installation…{C.RESET}")
    all_ok = verify_install(sys_info)

    # ── Done ─────────────────────────────────────────────────────────────────
    print(f"""
{C.GREEN}{C.BOLD}╔══════════════════════════════════════════════════════════╗
║                  Setup Complete!  🎉                     ║
╚══════════════════════════════════════════════════════════╝{C.RESET}
""")
    if not all_ok:
        warn("Some packages had issues. The app may still work — check warnings above.")

    # Return the venv python path for use by the launchers
    return str(_venv_python())


if __name__ == "__main__":
    setup()
