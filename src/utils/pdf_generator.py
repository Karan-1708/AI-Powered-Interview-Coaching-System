import os
import markdown
import re
from fpdf import FPDF
from src.utils.diagnostics import get_logger, safe_execute

logger = get_logger()

class PDFGenerator:
    @staticmethod
    @safe_execute(default_val="", log_msg="Unicode Cleaning Error")
    def clean_unicode(text):
        """
        Standardizes text for PDF generation.
        Replaces problematic Unicode characters with standard ASCII equivalents.
        """
        if not text:
            return ""
        
        replacements = {
            '\u2018': "'", '\u2019': "'", '\u201c': '"', '\u201d': '"',
            '\u2013': "-", '\u2014': "--", '\u2026': "...", '\u00a0': " ",
            '\u2022': "*", '\u2043': "*", '\u2012': "-", '\u2015': "--"
        }
        
        for unicode_char, ascii_char in replacements.items():
            text = text.replace(unicode_char, ascii_char)
            
        # Strip characters that are not in the Latin-1 range to prevent FPDF errors
        return text.encode('latin-1', 'replace').decode('latin-1')

    @staticmethod
    @safe_execute(default_val=False, log_msg="PDF Report Generation Error")
    def generate_report(job_title, industry, metrics_data, final_feedback, full_transcript, output_path):
        """
        Generates a high-quality, formatted PDF report using HTML rendering for Markdown support.
        """
        try:
            # 1. Clean and Sanitize Input
            final_feedback = str(final_feedback).strip()
            full_transcript = str(full_transcript).strip()
            
            # Standardize Unicode for Latin-1 compatibility
            final_feedback = PDFGenerator.clean_unicode(final_feedback)
            full_transcript = PDFGenerator.clean_unicode(full_transcript)

            # 2. Convert LLM Markdown to HTML
            feedback_html = markdown.markdown(final_feedback, extensions=['extra']).strip()
            feedback_html = feedback_html.replace("<strong>", "<b>").replace("</strong>", "</b>")
            feedback_html = feedback_html.replace("<h3>", '<h3 style="color: #2980b9;">').replace("<h2>", '<h2 style="color: #2c3e50;">')

            # Initialize PDF with a custom footer
            class PDF(FPDF):
                def footer(self):
                    # Position at 1.5 cm from bottom
                    self.set_y(-15)
                    self.set_font("helvetica", "I", 8)
                    self.set_text_color(150, 150, 150)
                    # Page number
                    self.cell(0, 10, f"Built with Data Drifters AI Coaching System - Page {self.page_no()}", align="C")

            pdf = PDF()
            pdf.set_auto_page_break(auto=True, margin=20)
            pdf.add_page()
            
            # --- HEADER ---
            pdf.set_font("helvetica", "B", 22)
            pdf.set_text_color(41, 128, 185) # Professional Blue
            pdf.cell(0, 15, "AI Interview Coaching Report", ln=True, align="C")
            
            pdf.set_font("helvetica", "I", 12)
            pdf.set_text_color(100, 100, 100)
            pdf.cell(0, 8, f"Role: {job_title.title()} | Industry: {industry.title()}", ln=True, align="C")
            pdf.ln(10)
            
            # --- METRICS BOX ---
            pdf.set_fill_color(240, 242, 246)
            pdf.set_font("helvetica", "B", 14)
            pdf.set_text_color(44, 62, 80)
            pdf.cell(0, 10, "Acoustic Metrics Summary", ln=True, fill=True)
            
            pdf.set_font("helvetica", "", 11)
            pdf.set_text_color(0, 0, 0)
            pdf.set_x(pdf.l_margin)
            metrics_summary = f"""
            <ul>
                <li><b>Average Pacing:</b> {metrics_data['wpm']:.0f} WPM (Ideal Range: 130-160)</li>
                <li><b>Total Filler Words:</b> {metrics_data['fillers']} (Um, Uh, Like)</li>
                <li><b>Awkward Pauses:</b> {metrics_data.get('pauses', 0)} (Silences > 1.5s)</li>
                <li><b>Stutters/Blunders:</b> {metrics_data.get('blunders', 0)} (Repetitions/Corrections)</li>
                <li><b>Dominant Tone:</b> {metrics_data.get('tone', 'Neutral')}</li>
                <li><b>Total Speaking Time:</b> {metrics_data['duration']:.1f} Seconds</li>
            </ul>
            """
            pdf.write_html(metrics_summary)
            pdf.ln(5)
            
            # --- AI FEEDBACK SECTION ---
            pdf.set_font("helvetica", "B", 14)
            pdf.set_fill_color(232, 244, 248)
            pdf.cell(0, 10, "AI Coach Evaluation", ln=True, fill=True)
            pdf.ln(2)
            
            pdf.set_font("helvetica", "", 11)
            pdf.set_x(pdf.l_margin)
            pdf.write_html(feedback_html)
            
            # --- TRANSCRIPT SECTION ---
            # Start transcript on a new page for clarity
            pdf.add_page()
            pdf.set_font("helvetica", "B", 14)
            pdf.set_fill_color(245, 245, 245)
            pdf.cell(0, 10, "Session Transcript", ln=True, fill=True)
            pdf.ln(5)
            
            pdf.set_font("helvetica", "", 10)
            pdf.set_x(pdf.l_margin)
            pdf.write_html(f'<div style="font-size: 10pt; line-height: 1.5;">{full_transcript}</div>')

            # 3. Final Save
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            pdf.output(output_path)
            return True
            
        except Exception as e:
            logger.error(f"Enterprise PDF Generation Failed: {e}", exc_info=True)
            return False
