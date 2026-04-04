import json
import re
import ast
import logging

logger = logging.getLogger("AI_Coach")

def clean_llm_text(text):
    """Deep cleans LLM output from all formatting artifacts (JSON, lists, numbering)."""
    raw = str(text).strip()
    
    # 1. Strip all leading/trailing brackets, quotes, and whitespace iteratively
    while raw and (raw[0] in '["\' \n\r\t' or raw[-1] in ']"\' \n\r\t'):
        raw = raw.strip('[]"\' \n\r\t')
    
    # 2. Global numbering removal (e.g., "1. ", "2) ", etc.) at starts of lines
    raw = re.sub(r'^\d+[\.\)]\s*', '', raw)
    raw = re.sub(r'\n\d+[\.\)]\s*', '\n', raw)
    
    # 3. Specific internal cleanup for concatenated list fragments like "] ["
    raw = raw.replace('"] ["', ' ').replace('] [', ' ')
    
    # 4. Remove specific prefixes
    raw = re.sub(r'^(Question\s*\d+:)\s*', '', raw, flags=re.IGNORECASE)
    
    return raw.strip()

def parse_file(uploaded_file):
    """Extracts text from PDF or TXT files."""
    if not uploaded_file: return None
    try:
        if uploaded_file.type == "text/plain":
            return str(uploaded_file.read(), "utf-8")
        elif uploaded_file.type == "application/pdf":
            try:
                import pypdf
                pdf = pypdf.PdfReader(uploaded_file)
                return " ".join([page.extract_text() for page in pdf.pages])
            except ImportError:
                return "Error: 'pypdf' library missing. PDF extraction disabled."
    except Exception as e:
        return f"Error parsing file: {e}"
    return None
