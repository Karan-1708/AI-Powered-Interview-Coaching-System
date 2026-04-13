import streamlit as st
import pandas as pd
import plotly.express as px
import os
import time
from src.utils.file_manager import FileManager
from src.utils.pdf_generator import PDFGenerator
from src.utils.history import HistoryManager

def render_final_analysis(session_data):
    """Renders the multi-tab final analysis and PDF export."""
    st.divider()
    st.header("📊 Final Analysis")
    
    tab_feed, tab_met, tab_trans = st.tabs(["🧠 AI Feedback", "📈 Metrics", "📝 Transcript"])
    
    with tab_feed:
        st.markdown(session_data['final_feedback'])
        
    with tab_met:
        c1, c2, c3 = st.columns(3)
        c1.metric("Avg Pacing", f"{session_data['avg_wpm']:.0f} WPM")
        c2.metric("Total Fillers", session_data['total_fillers'])
        c3.metric("Speaking Time", f"{session_data['total_duration']:.1f}s")
        
    with tab_trans:
        st.markdown(session_data['full_transcript'], unsafe_allow_html=True)
    
    # PDF Export
    pdf_p = os.path.join(FileManager.TEMP_DIR, f"report_{int(time.time())}.pdf")
    if PDFGenerator.generate_report(
        session_data['job_title'], 
        session_data['industry'], 
        {'wpm': session_data['avg_wpm'], 'fillers': session_data['total_fillers'], 'duration': session_data['total_duration']}, 
        session_data['final_feedback'], 
        session_data['full_transcript'], 
        pdf_p
    ):
        with open(pdf_p, "rb") as f:
            st.download_button(
                "📄 Download PDF Report", 
                f.read(), 
                f"Report_{session_data['job_title']}.pdf", 
                "application/pdf", 
                width='stretch'
            )

def render_history_dashboard():
    """Renders the gamified coaching progression dashboard."""
    st.header("📈 Coaching Progression Dashboard")
    history = HistoryManager.load_history()
    
    if not history:
        st.info("No history yet. Complete your first interview to see analytics!")
        return

    df = pd.DataFrame(history)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # --- GAMIFICATION CARDS ---
    st.markdown("### 🏆 Performance Badges")
    c1, c2, c3, c4 = st.columns(4)
    
    total_sessions = len(df)
    avg_wpm_all = df['wpm'].mean()
    total_fillers_all = df['fillers'].sum()
    
    with c1:
        if total_sessions >= 10: st.success("🎓 Master Interviewer")
        elif total_sessions >= 5: st.info("🏅 Seasoned Candidate")
        else: st.info("🌱 Aspiring Professional")
        st.caption("**Experience:** Based on your total completed sessions. 10+ for Master status.")
    
    with c2:
        if 130 <= avg_wpm_all <= 160: st.success("🎯 Golden Pacer")
        else: st.warning("🐢 Developing Rhythm")
        st.caption("**Clarity:** Awarded for maintaining an ideal professional pace (130-160 WPM).")
    
    with c3:
        if total_fillers_all < total_sessions * 2: st.success("✨ Silver Tongue")
        else: st.info("🗣️ Work in Progress")
        st.caption("**Eloquence:** Aim for fewer than 2 filler words per session on average.")
    
    with c4:
        st.info(f"🔥 {total_sessions} Sessions")
        st.caption("**Consistency:** Your total volume of training. Keep the streak alive!")

    # --- PROGRESSION CHARTS ---
    st.divider()
    st.markdown("### 📊 Pacing & Clarity Trends")
    
    fig_wpm = px.line(df, x='timestamp', y='wpm', title='WPM Progression',
                    markers=True, color_discrete_sequence=['#2980b9'])
    fig_wpm.add_hline(y=130, line_dash="dash", line_color="green", annotation_text="Ideal Start")
    fig_wpm.add_hline(y=160, line_dash="dash", line_color="green", annotation_text="Ideal End")
    st.plotly_chart(fig_wpm, width='stretch')
    
    fig_fill = px.bar(df, x='timestamp', y='fillers', title='Filler Word Count per Session',
                    color_discrete_sequence=['#e74c3c'])
    st.plotly_chart(fig_fill, width='stretch')

    with st.expander("📝 Detailed Session Logs"):
        st.dataframe(df.sort_values(by='timestamp', ascending=False), width='stretch')