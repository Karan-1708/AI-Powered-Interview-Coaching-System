import os
import markdown
try:
    from PIL import Image as PILImage
except ImportError:
    pass
from fpdf import FPDF

class PDFGenerator:
    @staticmethod
    def clean_unicode(text):
        """
        Replaces 'smart' quotes and other non-latin-1 characters 
        with standard PDF-safe equivalents.
        """
        if not text:
            return ""
        
        # Mapping of common 'smart' characters to standard equivalents
        replacements = {
            '\u2018': "'",  # Left single quote
            '\u2019': "'",  # Right single quote
            '\u201c': '"',  # Left double quote
            '\u201d': '"',  # Right double quote
            '\u2013': "-",  # En dash
            '\u2014': "--", # Em dash
            '\u2026': "...", # Ellipsis
        }
        
        for unicode_char, ascii_char in replacements.items():
            text = text.replace(unicode_char, ascii_char)
            
        # Last resort: encode to latin-1 and ignore anything that fails
        return text.encode('latin-1', 'ignore').decode('latin-1')

    @staticmethod
    def generate_report(job_title, industry, metrics_data, final_feedback, full_transcript, output_path):
        try:
            # 1. Sanitize all incoming LLM text immediately
            final_feedback = PDFGenerator.clean_unicode(final_feedback)
            full_transcript = PDFGenerator.clean_unicode(full_transcript)

            pdf = FPDF()
            pdf.add_page()
            pdf.set_auto_page_break(auto=True, margin=15)
            
            # --- TITLE PAGE HEADER ---
            pdf.set_font("helvetica", "B", 22)
            pdf.set_text_color(41, 128, 185) # Professional Blue
            pdf.cell(0, 10, "Data Drifters: AI Interview Report", ln=True, align="C")
            
            pdf.set_font("helvetica", "I", 12)
            pdf.set_text_color(100, 100, 100)
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
            feedback_html = markdown.markdown(final_feedback)
            # fpdf2 prefers <b> over <strong>
            feedback_html = feedback_html.replace("<strong>", "<b>").replace("</strong>", "</b>")
            
            # --- RENDER FEEDBACK ---
            feedback_section = f"""
            <h2 style="color: #2c3e50;">AI Coach Feedback</h2>
            {feedback_html}
            """
            pdf.write_html(feedback_section)
            
            # --- RENDER TRANSCRIPT ---
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