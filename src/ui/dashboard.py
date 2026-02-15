import streamlit as st  # pyright: ignore[reportMissingImports]

def render_dashboard(transcript, metrics, duration, selected_mode, tier):
    """
    Renders the Analysis Results column (Right Side).
    """
    st.subheader("2. Analysis Results")
    
    # 1. Metrics Grid
    feedback = metrics["feedback"]
    
    # Create 5 columns for the metrics
    m1, m2, m3, m4, m5 = st.columns(5)
    
    m1.metric("Tone", metrics["tone_label"], feedback["tone"], delta_color=feedback["tone_status"])
    m2.metric("Speed", f"{metrics['wpm']} WPM", feedback["wpm"], delta_color=feedback["wpm_status"])
    m3.metric("Pauses", f"{metrics['pause_count']}", feedback["pause"], delta_color=feedback["pause_status"])
    m4.metric("Fillers", f"{metrics['filler_count']}", feedback["filler"], delta_color=feedback["filler_status"])
    m5.metric("Blunders", f"{metrics['blunder_count']}", feedback["blunder"], delta_color=feedback["blunder_status"])

    st.divider()

    # 2. Coach Tip (Soft Logic)
    if "density_tip" in feedback:
        st.info(f"💡 **Coach Tip:** {feedback['density_tip']}")

    # 3. Transcript
    st.markdown("### 📝 Transcript")
    st.write(transcript)

    # 4. Technical Debug Stats (Collapsible)
    with st.expander("System Stats (Debug)"):
        st.write(f"**Mode:** {selected_mode}")
        st.write(f"**Hardware Tier:** {tier}")
        st.write(f"**Processing Time:** {duration:.2f}s")
        if "energy_avg" in metrics:
            st.write(f"**Energy (Vol):** {metrics['energy_avg']}")
        if "pitch_var" in metrics:
            st.write(f"**Pitch Var:** {metrics['pitch_var']}")