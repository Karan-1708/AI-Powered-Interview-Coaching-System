import os
import platform
import shutil
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from typing import List, Optional

# --- 1. OMP & NVIDIA DLL FIXES (MUST BE BEFORE ML IMPORTS) ---
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

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

# --- 2. NOW IMPORT YOUR BACKEND ENGINES ---
from src.backend.audio_processor import AudioProcessor
from src.backend.llm_client import LLMClient
from src.utils.diagnostics import get_logger
from src.utils.file_manager import FileManager

logger = get_logger()

# Initialize engines at the server level
processor = AudioProcessor()

app = FastAPI(title="Data Drifters: Interview Coach API")

# --- DATA MODELS (For clean request/response validation) ---
class ChatMessage(BaseModel):
    role: str
    content: str

class LLMRequest(BaseModel):
    system_prompt: str
    user_message: str
    chat_history: Optional[List[ChatMessage]] = []
    tier: str = "Balanced"

# --- 1. THE AUDIO PROCESSING ENDPOINT ---
@app.post("/process-audio")
async def process_audio(
    file: UploadFile = File(...), 
    difficulty: str = Form("Standard Interview"), 
    tier: str = Form("Balanced")
):
    """
    Receives an audio file, transcribes it using Whisper, 
    and calculates acoustic metrics.
    """
    # 1. Initialize safe directories and save the uploaded file
    FileManager.initialize_directories()
    temp_path = os.path.join(FileManager.TEMP_DIR, f"upload_{file.filename}")
    
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        # 2. Call your existing AudioProcessor logic
        transcript, metrics, duration, error = processor.process_interview(
            temp_path, difficulty=difficulty, tier=tier
        )
        
        if error:
            raise HTTPException(status_code=500, detail=error)
            
        # 3. Clean up the temp file safely via FileManager
        FileManager.safe_delete_file(temp_path)
        
        return {
            "transcript": transcript,
            "metrics": metrics,
            "duration": duration
        }
        
    except Exception as e:
        # Failsafe cleanup in case of a crash
        FileManager.safe_delete_file(temp_path)
        logger.error(f"API Audio Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- 2. THE LLM GENERATION ENDPOINT ---
@app.post("/generate-response")
async def generate_response(request: LLMRequest):
    """
    Communicates with the local Ollama server to get the next AI question.
    """
    try:
        # Dynamically select the model based on the tier
        llm = LLMClient(tier_string=request.tier)
        
        # Convert Pydantic objects back to a simple list of dicts for Ollama
        history = [msg.dict() for msg in request.chat_history] if request.chat_history else []
        
        response_text = llm.generate_response(
            system_prompt=request.system_prompt,
            user_message=request.user_message,
            chat_history=history
        )
        
        return {"response": response_text, "model_used": llm.model_name}
        
    except Exception as e:
        logger.error(f"API LLM Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.server:app", host="127.0.0.1", port=8000, reload=True)