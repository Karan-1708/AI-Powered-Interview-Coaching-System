import os
import markdown
from fpdf import FPDF

class PDFGenerator:
    @staticmethod
    def generate_report(job_title, industry, metrics_data, final_feedback, full_transcript, output_path):
        try:
            pdf = FPDF()
            pdf.add_page()
            pdf.set_auto_page_break(auto=True, margin=15)
            
            # --- TITLE PAGE HEADER ---
            pdf.set_font("helvetica", "B", 22)
            pdf.set_text_color(41, 128, 185) # Professional Blue
            pdf.cell(0, 10, "Data Drifters: AI Interview Report", ln=True, align="C")
            
            pdf.set_font("helvetica", "I", 12)
            pdf.set_text_color(100, 100, 100)
            # Capitalize the inputs so "helpdesk" becomes "Helpdesk"
            pdf.cell(0, 8, f"Target Role: {job_title.title()} | Industry: {industry.title()}", ln=True, align="C")
            pdf.ln(10)
            
            pdf.set_text_color(0, 0, 0)
            
            # --- METRICS DASHBOARD ---
            metrics_html = f"""
            <h2 style="color: #2c3e50;">Acoustic Metrics Summary</h2>
            <ul>
                <li><b>Average Pacing:</b> {metrics_data['wpm']:.0f} WPM <i>(Ideal: 130-160)</i></li>
                <li><b>Total Filler Words:</b> {metrics_data['fillers']}</li>
                <li><b>Total Speaking Time:</b> {metrics_data['duration']:.1f} Seconds</li>
            </ul>
            <hr>
            """
            pdf.write_html(metrics_html)
            pdf.ln(5)
            
            # --- PARSE LLM MARKDOWN ---
            # Convert Markdown to HTML, then replace strong tags with b tags for fpdf2 compatibility
            feedback_html = markdown.markdown(final_feedback)
            feedback_html = feedback_html.replace("<strong>", "<b>").replace("</strong>", "</b>")
            
            # We don't parse transcript_html through Markdown because we already hardcoded HTML into it in app.py!
            
            # --- RENDER FEEDBACK ---
            feedback_section = f"""
            <h2 style="color: #2c3e50;">AI Coach Feedback</h2>
            {feedback_html}
            """
            pdf.write_html(feedback_section)
            
            # --- RENDER TRANSCRIPT (On a new page to prevent awkward cutting) ---
            pdf.add_page()
            transcript_section = f"""
            <h2 style="color: #2c3e50;">Session Transcript</h2>
            <hr>
            <br>
            {full_transcript}
            """
            pdf.write_html(transcript_section)
            
            pdf.output(output_path)
            return True
            
        except Exception as e:
            from src.utils.diagnostics import get_logger
            logger = get_logger()
            logger.error(f"PDF Generation Failed: {e}", exc_info=True)
            return False