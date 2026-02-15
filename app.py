import os
import platform
import shutil
import streamlit as st  # pyright: ignore[reportMissingImports]
import logging
from src.ui.dashboard import render_dashboard

# --- 1. GLOBAL CRASH PROTECTION ---
if platform.system() == "Windows":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from src.ui.recorder import record_audio
from src.backend.audio_processor import AudioProcessor
from src.backend.hardware import HardwareInfo
from src.backend.monitor import ResourceMonitor
from src.utils.diagnostics import log_system_info, get_logger

# Initialize Logging
log_system_info()
logger = get_logger()

# --- DLL FIX ---
def register_nvidia_dlls():
    if platform.system() != "Windows": return
    try:
        import site
        possible_paths = site.getsitepackages()
        try: possible_paths.append(site.getusersitepackages())
        except: pass
        for base_path in possible_paths:
            cublas_bin = os.path.join(base_path, "nvidia", "cublas", "bin")
            cudnn_bin = os.path.join(base_path, "nvidia", "cudnn", "bin")
            if os.path.exists(os.path.join(cublas_bin, "cublas64_12.dll")):
                os.environ["PATH"] += os.pathsep + cublas_bin
                os.environ["PATH"] += os.pathsep + cudnn_bin
                if hasattr(os, "add_dll_directory"):
                    os.add_dll_directory(cublas_bin)
                    os.add_dll_directory(cudnn_bin)
                break
    except Exception: pass
register_nvidia_dlls()

# --- HELPER: PRIVACY CLEANUP ---
def cleanup_data():
    """Deletes all temporary files (Privacy Feature) - Windows Safe Version"""
    deleted_files = 0
    
    # 1. Clean Audio Files
    if os.path.exists("temp_data"):
        for f in os.listdir("temp_data"):
            try:
                os.remove(os.path.join("temp_data", f))
                deleted_files += 1
            except Exception:
                pass # Skip if file is actively being recorded

    # 2. Clean Log Files (Critical Fix for WinError 32)
    if os.path.exists("logs"):
        # Step A: Get the logger and find the file handler
        logger = logging.getLogger()
        
        # Step B: Close and remove all handlers to release the Windows file lock
        handlers = logger.handlers[:]
        for handler in handlers:
            handler.close()
            logger.removeHandler(handler)
        
        # Step C: Now it is safe to delete
        for f in os.listdir("logs"):
            try:
                os.remove(os.path.join("logs", f))
                deleted_files += 1
            except Exception as e:
                st.error(f"Could not delete {f}: {e}")
            
    return deleted_files

# --- MAIN APP ---
st.set_page_config(page_title="AI Interview Coach", page_icon="🎙️", layout="wide")

def main():
    try:
        hw = HardwareInfo()
        monitor = ResourceMonitor()
        
        st.title("🎙️ AI Interview Coach")
        
        # --- SIDEBAR ---
        st.sidebar.header("🖥️ Hardware Monitor")
        rec_tier, rec_reason = hw.get_recommendation()
        
        # Recommendation Logic
        default_index = 0
        if "Balanced" in rec_tier: default_index = 1
        if "Pro" in rec_tier: default_index = 2
        
        selected_tier = st.sidebar.selectbox("Performance Profile", 
            ["Eco (Low Spec)", "Balanced (Mid Spec)", "Pro (High Spec)"], 
            index=default_index,
            help=f"Recommendation: {rec_reason}"
        )

        # Resource Bars
        stats = monitor.get_system_usage()
        if "Pro" in selected_tier and hw.has_nvidia:
            st.sidebar.progress(stats['vram_percent'] / 100, text=f"VRAM: {stats['vram_used_gb']}/{stats['vram_total_gb']} GB")
        else:
            st.sidebar.progress(stats['cpu_percent'] / 100, text=f"CPU: {stats['cpu_percent']}%")
            st.sidebar.progress(stats['ram_percent'] / 100, text=f"RAM: {stats['ram_used_gb']}/{stats['ram_total_gb']} GB")

        st.sidebar.divider()

        # --- PRIVACY & DATA MANAGEMENT (NEW) ---
        st.sidebar.header("🔒 Privacy & Data")
        
        # Count files
        audio_files = len(os.listdir("temp_data")) if os.path.exists("temp_data") else 0
        log_files = len(os.listdir("logs")) if os.path.exists("logs") else 0
        total_files = audio_files + log_files
        
        st.sidebar.caption(f"Stored Data: {audio_files} recordings, {log_files} logs.")
        
        if total_files > 0:
            if st.sidebar.button("🗑️ Delete All Data", type="primary"):
                count = cleanup_data()
                st.sidebar.success(f"Deleted {count} files.")
                st.rerun()
        else:
            st.sidebar.info("System is clean.")

        st.sidebar.divider()

        # --- MAIN INTERFACE ---
        mode_options = ["Practice Mode", "Standard Interview", "Technical / Complex", "Presentation"]
        selected_mode = st.sidebar.selectbox("Analysis Mode", mode_options, index=1)
        
        processor = AudioProcessor()
        col1, col2 = st.columns([1, 2])

        with col1:
            st.subheader("1. Record Answer")
            audio_path = record_audio()
            
            if audio_path:
                st.audio(audio_path)
                btn_label = f"Analyze"
                
                if st.button(btn_label, type="primary", use_container_width=True):
                    with st.spinner("Analyzing..."):
                        transcript, metrics, duration, error = processor.process_interview(
                            audio_path, difficulty=selected_mode, tier=selected_tier
                        )
                    
                    if error:
                        st.error(f"⚠️ {error}")
                        logger.error(f"User Error: {error}")
                    else:
                        st.session_state['results'] = (transcript, metrics, duration)

        # --- RIGHT COLUMN: Results ---
        with col2:
            if 'results' in st.session_state:
                # Unpack the data
                transcript, metrics, duration = st.session_state['results']
                
                # Call the new module to draw everything
                render_dashboard(
                    transcript, 
                    metrics, 
                    duration, 
                    selected_mode, 
                    selected_tier
                )
            else:
                st.info("Ready to analyze. Select settings and record your answer.")

    except Exception as e:
        st.error("🚨 An unexpected error occurred.")
        st.code(str(e))
        logger.critical(f"Global Crash: {e}", exc_info=True)

if __name__ == "__main__":
    os.makedirs("temp_data", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    main()