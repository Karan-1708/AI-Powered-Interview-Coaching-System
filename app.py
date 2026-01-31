import os
import sys
import site

# --- CONFIGURATION / DLL FIX START ---
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def register_nvidia_dlls():
    if sys.platform != "win32": return
    try:
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
# --- DLL FIX END ---

import streamlit as st
from src.ui.recorder import record_audio
from src.backend.audio_processor import AudioProcessor

st.set_page_config(page_title="AI Interview Coach", page_icon="🎙️", layout="wide")

def main():
    st.title("🎙️ AI Interview Coach")
    st.caption("Phase 3: Acoustic Analysis Engine (WPM, Tone & Pauses)")

    # --- Sidebar Settings (UPDATED) ---
    st.sidebar.header("⚙️ Context Settings")
    
    # New options based on your prompt
    mode_options = [
        "Practice Mode", 
        "Standard Interview", 
        "Technical / Complex", 
        "Presentation"
    ]
    
    selected_mode = st.sidebar.selectbox(
        "Analysis Mode",
        mode_options,
        index=1,
        help="Adjusts WPM targets based on the type of question."
    )
    
    # Dynamic Explanation in Sidebar
    if selected_mode == "Standard Interview":
        st.sidebar.success("Target: **140–160 WPM**\n\nIdeal for most behavioral questions. Projects confidence and clarity.")
    elif selected_mode == "Technical / Complex":
        st.sidebar.info("Target: **100–130 WPM**\n\nBest for complex answers where clarity is more critical than speed.")
    elif selected_mode == "Presentation":
        st.sidebar.warning("Target: **130–150 WPM**\n\nDeliberate pacing for engaging an audience.")
    else:
        st.sidebar.write("Target: **100–170 WPM**\n\nWide range for general practice.")

    try:
        processor = AudioProcessor()
        col1, col2 = st.columns([1, 2])

        # --- LEFT COLUMN ---
        with col1:
            st.subheader("1. Record Answer")
            audio_path = record_audio()
            
            if audio_path:
                st.audio(audio_path)
                
                if st.button("Analyze Performance", type="primary"):
                    with st.spinner(f"Analyzing for '{selected_mode}'..."):
                        # Pass selected mode to processor
                        transcript, metrics, duration, error = processor.process_interview(audio_path, selected_mode)
                    
                    if error:
                        st.error(f"⚠️ {error}")
                    else:
                        st.session_state['results'] = (transcript, metrics, duration)

        # --- RIGHT COLUMN ---
        # --- RIGHT COLUMN ---
        with col2:
            st.subheader("2. Analysis Results")
            
            if 'results' in st.session_state:
                transcript, metrics, duration = st.session_state['results']
                feedback = metrics["feedback"]
                
                # --- Metrics Grid ---
                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("Tone", metrics["tone_label"], feedback["tone"], delta_color=feedback["tone_status"])
                m2.metric("Speed (WPM)", f"{metrics['wpm']}", feedback["wpm"], delta_color=feedback["wpm_status"])
                m3.metric("Pauses", f"{metrics['pause_count']}", feedback["pause"], delta_color=feedback["pause_status"])
                m4.metric("Fillers", f"{metrics['filler_count']}", feedback["filler"], delta_color=feedback["filler_status"])
                m5.metric("Blunders", f"{metrics['blunder_count']}", feedback["blunder"], delta_color=feedback["blunder_status"])

                st.divider()
                
                # --- SOFT LOGIC TIP (Dynamic) ---
                if "density_tip" in feedback:
                    st.info(f"💡 **Content Tip:** {feedback['density_tip']}")

                with st.expander("Technical Audio Stats (Debug)"):
                    st.write(f"**Duration:** {metrics['duration']}s")
                    st.write(f"**Target Range:** {selected_mode}")
                    st.write(f"**Energy:** {metrics['energy_avg']}")

                st.markdown("### 📝 Transcript")
                st.write(transcript)
                
            else:
                st.info("Select a mode from the sidebar, record, and analyze.")

    except Exception as e:
        st.error("🚨 Critical Application Error")
        st.code(str(e))

if __name__ == "__main__":
    os.makedirs("temp_data", exist_ok=True)
    main()