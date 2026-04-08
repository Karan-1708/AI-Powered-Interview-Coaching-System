import os
import sys
import platform
import subprocess
import shutil

def run_command(command):
    """Executes a shell command and prints output."""
    print(f"Executing: {command}")
    try:
        subprocess.check_call(command, shell=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e}")
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

    # 2. Upgrade Pip
    print("\n[1/3] Upgrading pip...")
    run_command(f'"{sys.executable}" -m pip install --upgrade pip')

    # 3. Install Optimized AI Engine (Torch)
    print("\n[2/3] Installing optimized AI Engine (Torch)...")
    
    if os_name == "Windows":
        if sys.version_info >= (3, 13):
            print("[INFO] Python 3.13 detected. Installing Nightly CUDA 12.4...")
            run_command(f'"{sys.executable}" -m pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu124')
        else:
            print("[INFO] Python 3.10-3.12 detected. Installing Stable CUDA 12.1...")
            run_command(f'"{sys.executable}" -m pip install torch --index-url https://download.pytorch.org/whl/cu121')
            
    elif os_name == "Darwin":  # macOS
        if "arm" in machine or "64" in machine:
            print("[INFO] Apple Silicon/Intel detected. Installing Torch with Metal/MPS support...")
            run_command(f'"{sys.executable}" -m pip install torch')
            
    elif os_name == "Linux":
        if sys.version_info >= (3, 13):
            print("[INFO] Linux + Python 3.13 detected. Installing Nightly CUDA 12.4...")
            run_command(f'"{sys.executable}" -m pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu124')
        else:
            print("[INFO] Linux detected. Installing Stable CUDA 12.1...")
            run_command(f'"{sys.executable}" -m pip install torch --index-url https://download.pytorch.org/whl/cu121')

    # 4. Install remaining dependencies
    print("\n[3/3] Installing remaining dependencies from requirements.txt...")
    if os.path.exists("requirements.txt"):
        run_command(f'"{sys.executable}" -m pip install -r requirements.txt')
    else:
        print("[ERROR] requirements.txt not found!")

    # 5. Setup .env
    if not os.path.exists(".env") and os.path.exists(".env.example"):
        print("\n[INFO] Creating .env file from .env.example...")
        shutil.copy(".env.example", ".env")

    print("\n===================================================")
    print("✅ Setup Complete!")
    print("You can now run the application:")
    print("  Backend: python -m uvicorn src.api.server:app --reload")
    print("  Frontend: streamlit run app.py")
    print("===================================================")

if __name__ == "__main__":
    setup()
