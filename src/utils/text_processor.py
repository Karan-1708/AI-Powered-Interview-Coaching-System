import re
from src.utils.diagnostics import get_logger

logger = get_logger()

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
    """Extracts text from PDF, TXT, or DOCX files."""
    if not uploaded_file: return None
    
    try:
        # 1. Text Files
        if uploaded_file.type == "text/plain":
            return str(uploaded_file.read(), "utf-8")
            
        # 2. PDF Files (using pdfplumber)
        elif uploaded_file.type == "application/pdf":
            import pdfplumber
            text = ""
            with pdfplumber.open(uploaded_file) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            return text.strip() if text.strip() else None
            
        # 3. Word Documents (using python-docx)
        elif uploaded_file.type in ["application/vnd.openxmlformats-officedocument.wordprocessingml.document", "application/msword"]:
            from docx import Document
            doc = Document(uploaded_file)
            text = "\n".join([para.text for para in doc.paragraphs])
            return text.strip() if text.strip() else None
            
    except Exception as e:
        logger.error(f"Text extraction failed for {uploaded_file.name}: {e}")
        return None # Return None to trigger fallback in UI
        
    return None
