import os
import sys
import platform
import subprocess
import shutil
import time

def run_command(command):
    """Executes a shell command and prints output."""
    print(f"Executing: {command}")
    try:
        subprocess.check_call(command, shell=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e}")
        return False

def download_ffmpeg_windows():
    """Downloads and extracts a portable version of FFmpeg for Windows."""
    if os.path.exists("bin/ffmpeg.exe"):
        print("[INFO] Portable FFmpeg already exists in bin/")
        return True

    print("[INFO] FFmpeg not found. Downloading portable version for Windows...")
    try:
        import urllib.request
        import zipfile
        
        # Ensure bin directory exists
        os.makedirs("bin", exist_ok=True)
        
        url = "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip"
        zip_path = "ffmpeg.zip"
        
        print("[INFO] Fetching FFmpeg from GitHub...")
        urllib.request.urlretrieve(url, zip_path)
        
        print("[INFO] Extracting FFmpeg...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Find the exe files in the zip and extract only them
            for member in zip_ref.namelist():
                if member.endswith("ffmpeg.exe") or member.endswith("ffprobe.exe"):
                    filename = os.path.basename(member)
                    with zip_ref.open(member) as source, open(os.path.join("bin", filename), "wb") as target:
                        shutil.copyfileobj(source, target)
        
        # Cleanup
        os.remove(zip_path)
        print("[SUCCESS] Portable FFmpeg ready in bin/")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to download FFmpeg: {e}")
        return False

def setup():
    print("===================================================")
    print("  AI Interview Coach - Developer Setup Script")
    print("===================================================")

    # 1. Detect Environment
    os_name = platform.system()
    py_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    machine = platform.machine().lower()
    
    print(f"[INFO] OS: {os_name}")
    print(f"[INFO] Python Version: {py_version}")
    print(f"[INFO] Architecture: {machine}")

    # 2. System Dependencies (FFmpeg)
    print("\n[1/5] Checking System Dependencies (FFmpeg)...")
    if os_name == "Windows":
        download_ffmpeg_windows()
    elif os_name == "Darwin": # macOS
        if not shutil.which("ffmpeg"):
            print("[INFO] FFmpeg missing. Attempting install via brew...")
            run_command("brew install ffmpeg")
    elif os_name == "Linux":
        if not shutil.which("ffmpeg"):
            print("[INFO] FFmpeg missing. Attempting install via apt...")
            run_command("sudo apt update && sudo apt install -y ffmpeg")

    # 3. Upgrade Pip
    print("\n[2/5] Upgrading pip...")
    run_command(f'"{sys.executable}" -m pip install --upgrade pip')

    # 4. Install Optimized AI Engine (Torch)
    print("\n[3/5] Cleaning and Installing optimized AI Engine (Torch)...")
    run_command(f'"{sys.executable}" -m pip uninstall torch torchaudio torchvision -y')
    
    if os_name == "Windows":
        if sys.version_info >= (3, 13):
            print("[INFO] Python 3.13 detected. Installing Nightly CUDA 12.4...")
            run_command(f'"{sys.executable}" -m pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu124 --force-reinstall --no-cache-dir')
        else:
            print("[INFO] Python 3.10-3.12 detected. Installing Stable CUDA 12.1...")
            run_command(f'"{sys.executable}" -m pip install torch --index-url https://download.pytorch.org/whl/cu121 --force-reinstall --no-cache-dir')
            
    elif os_name == "Darwin":
        print("[INFO] macOS detected. Installing Torch with Metal/MPS support...")
        run_command(f'"{sys.executable}" -m pip install torch --force-reinstall')
            
    elif os_name == "Linux":
        if sys.version_info >= (3, 13):
            run_command(f'"{sys.executable}" -m pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu124 --force-reinstall')
        else:
            run_command(f'"{sys.executable}" -m pip install torch --index-url https://download.pytorch.org/whl/cu121 --force-reinstall')

    # 5. Install remaining dependencies
    print("\n[4/5] Installing remaining dependencies from requirements.txt...")
    if os.path.exists("requirements.txt"):
        # Pre-install numpy 2.x for 3.13 stability
        run_command(f'"{sys.executable}" -m pip install "numpy>=2.1.0"')
        run_command(f'"{sys.executable}" -m pip install -r requirements.txt')
    else:
        print("[ERROR] requirements.txt not found!")

    # 6. Setup .env
    print("\n[5/5] Finalizing configuration...")
    if not os.path.exists(".env") and os.path.exists(".env.example"):
        print("[INFO] Creating .env file from .env.example...")
        shutil.copy(".env.example", ".env")

    print("\n===================================================")
    print("✅ Setup Complete!")
    print("Please RESTART your IDE terminal and then run:")
    print("  Backend: python -m uvicorn src.api.server:app --reload")
    print("  Frontend: streamlit run app.py")
    print("===================================================")

if __name__ == "__main__":
    setup()
