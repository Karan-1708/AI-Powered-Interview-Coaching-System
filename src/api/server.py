import os
import shutil
import requests
import json
import time
from dotenv import load_dotenv

# Load .env file if it exists
load_dotenv()

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from fastapi.responses import StreamingResponse, FileResponse
import edge_tts
from src.utils.diagnostics import get_logger, log_system_info, safe_execute

# --- 1. ENVIRONMENT & SECURITY SETUP ---
log_system_info()
logger = get_logger()

# Strict security: No hardcoded fallback. Must be set in .env or environment.
INTERNAL_API_KEY = os.getenv("INTERNAL_API_KEY")

def verify_internal_key(x_internal_key: str = Header(None)):
    """Dependency to verify the internal API key for inter-service security."""
    if not INTERNAL_API_KEY:
        logger.critical("SECURITY ALERT: INTERNAL_API_KEY is not set in environment!")
        raise HTTPException(status_code=500, detail="Server security misconfiguration")
        
    if not x_internal_key or x_internal_key != INTERNAL_API_KEY:
        logger.warning(f"Unauthorized access attempt with key: {x_internal_key}")
        raise HTTPException(status_code=401, detail="Invalid Internal API Key")
    return x_internal_key

# --- 2. BACKEND ENGINES ---
from src.backend.audio_processor import AudioProcessor
from src.backend.llm_client import LLMClient
from src.utils.file_manager import FileManager
from src.backend.hardware import HardwareInfo
from src.backend.monitor import ResourceMonitor

# Initialize engines at the server level
processor = AudioProcessor()
hw_info = HardwareInfo()
res_monitor = ResourceMonitor()

app = FastAPI(
    title="Data Drifters: Interview Coach API",
    description="Backend API for AI-powered interview coaching.",
    version="1.2.0"
)

# --- 3. CORS CONFIGURATION ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- DATA MODELS ---
class ChatMessage(BaseModel):
    role: str
    content: str

class LLMRequest(BaseModel):
    system_prompt: str
    user_message: str
    chat_history: Optional[List[ChatMessage]] = []
    provider: str
    model: str
    compute_type: str
    api_key: Optional[str] = None
    resume_context: Optional[str] = ""
    job_context: Optional[str] = ""

class QuestionRequest(BaseModel):
    seniority: str
    job_title: str
    industry: str
    selected_round: str
    engine_config: dict
    resume_context: Optional[str] = ""
    job_context: Optional[str] = ""

class ConnectionRequest(BaseModel):
    provider: str
    model: str
    compute_type: str
    api_key: Optional[str] = None

class SpeechRequest(BaseModel):
    text: str
    voice: Optional[str] = "en-US-GuyNeural"

# --- ENDPOINTS ---

@app.get("/", tags=["Health"])
def root():
    return {"status": "online", "message": "Data Drifters API is running."}

@app.get("/hardware", dependencies=[Depends(verify_internal_key)], tags=["Telemetry"])
def get_hardware():
    try:
        rec_tier, rec_reason = hw_info.get_recommendation()
        stats = res_monitor.get_system_usage()
        
        # Identify the physical hardware detected for the "Actually Using" indicator
        detected_hw = "Standard CPU"
        if hw_info.has_nvidia: detected_hw = "NVIDIA GPU"
        elif hw_info.is_apple_silicon: detected_hw = "Apple Silicon (M-Series)"
        
        return {
            "tier": rec_tier,
            "reason": rec_reason,
            "has_nvidia": hw_info.has_nvidia,
            "is_apple_silicon": hw_info.is_apple_silicon,
            "detected_hw": detected_hw,
            "stats": stats
        }
    except Exception as e:
        logger.error(f"Hardware API Error: {e}")
        raise HTTPException(status_code=500, detail="Hardware telemetry failed")

@app.post("/process-audio", dependencies=[Depends(verify_internal_key)], tags=["AI Processing"])
def process_audio(
    file: UploadFile = File(...), 
    difficulty: str = Form("Standard Interview"), 
    tier: str = Form("Balanced")
):
    FileManager.initialize_directories()
    temp_path = os.path.join(FileManager.TEMP_DIR, f"upload_{file.filename}")
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        transcript, metrics, duration, error = processor.process_interview(
            temp_path, difficulty=difficulty, tier=tier
        )
        if error:
            raise HTTPException(status_code=500, detail=error)
        return {"transcript": transcript, "metrics": metrics, "duration": duration}
    except HTTPException: raise
    except Exception as e:
        logger.error(f"API Audio error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        FileManager.safe_delete_file(temp_path)

@app.post("/test-connection", dependencies=[Depends(verify_internal_key)], tags=["LLM"])
def test_connection(request: ConnectionRequest):
    try:
        llm = LLMClient(provider=request.provider, model_name=request.model, 
                        compute_type=request.compute_type, api_key=request.api_key)
        success, message = llm.test_connection()
        return {"success": success, "message": message}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-response", dependencies=[Depends(verify_internal_key)], tags=["LLM"])
def generate_response(request: LLMRequest):
    """Generates LLM response with optional resume/job context."""
    try:
        llm = LLMClient(provider=request.provider, model_name=request.model, 
                        compute_type=request.compute_type, api_key=request.api_key)
        
        # Build contextual instruction
        context_instr = ""
        if request.resume_context:
            context_instr += f"\n[CONTEXT: RESUME]\n{request.resume_context}\n"
        if request.job_context:
            context_instr += f"\n[CONTEXT: JOB DESCRIPTION]\n{request.job_context}\n"
            
        full_system_prompt = request.system_prompt + context_instr
        
        history = [msg.dict() for msg in request.chat_history] if request.chat_history else []
        response_text = llm.generate_response(system_prompt=full_system_prompt, 
                                            user_message=request.user_message, chat_history=history)
        return {"response": response_text, "model_used": llm.model_name}
    except Exception as e:
        logger.error(f"Generate Response Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-questions", dependencies=[Depends(verify_internal_key)], tags=["LLM"])
def generate_questions(request: QuestionRequest):
    """Specifically generates interview questions using full role and context metadata."""
    try:
        config = request.engine_config
        llm = LLMClient(provider=config['provider'], model_name=config['model'], 
                        compute_type=config['compute'], api_key=config.get('api_key'))
        
        context_bonus = ""
        if request.resume_context:
            context_bonus += f"\nCandidate Resume Context: {request.resume_context[:2000]}"
        if request.job_context:
            context_bonus += f"\nJob Description Context: {request.job_context[:2000]}"

        q_prompt = (
            f"Generate 3 highly specific interview questions for a {request.seniority} {request.job_title} "
            f"during the '{request.selected_round}' round in the {request.industry} industry. {context_bonus} "
            f"Output each question on its own new line. Do NOT use numbers, bullet points, brackets, or quotes. Output ONLY the questions."
        )
        
        response = llm.generate_response(
            system_prompt="You are an expert interviewer. You output ONLY the plain text of your questions, one per line.", 
            user_message=q_prompt, 
            chat_history=[]
        )
        return {"response": response}
    except Exception as e:
        logger.error(f"Generate Questions Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models", dependencies=[Depends(verify_internal_key)], tags=["Local Models"])
def list_models():
    ollama_hosts = []
    env_host = os.getenv("OLLAMA_HOST")
    if env_host: ollama_hosts.append(env_host)
    ollama_hosts.extend(["http://host.docker.internal:11434", "http://127.0.0.1:11434", "http://localhost:11434"])
    for host in ollama_hosts:
        try:
            res = requests.get(f"{host}/api/tags", timeout=2)
            if res.status_code == 200:
                return {"models": [m['name'] for m in res.json().get('models', [])]}
        except: continue
    return {"models": []}

@app.post("/pull-model", dependencies=[Depends(verify_internal_key)], tags=["Local Models"])
def pull_model(request: dict):
    model_name = request.get("model")
    ollama_host = "http://host.docker.internal:11434"
    for host in [os.getenv("OLLAMA_HOST"), "http://host.docker.internal:11434", "http://127.0.0.1:11434"]:
        try:
            if requests.get(f"{host}/api/tags", timeout=1).status_code == 200:
                ollama_host = host
                break
        except: continue
    def generate():
        try:
            with requests.post(f"{ollama_host}/api/pull", json={"name": model_name}, 
                               stream=True, timeout=None) as r:
                for chunk in r.iter_lines():
                    if chunk: yield chunk.decode('utf-8') + "\n"
        except Exception as e:
            yield json.dumps({"error": str(e)}) + "\n"
    return StreamingResponse(generate(), media_type="application/x-ndjson")

@app.post("/generate-speech", dependencies=[Depends(verify_internal_key)], tags=["TTS"])
async def generate_speech(request: SpeechRequest):
    """
    Generates an MP3 file from text using edge-tts with safety limits.
    """
    # 1. Text Length Limiter (1500 characters)
    max_chars = 1500
    safe_text = request.text.strip()
    
    if len(safe_text) > max_chars:
        logger.warning(f"TTS input too long ({len(safe_text)} chars). Truncating...")
        # Truncate at nearest sentence boundary (. ! ?)
        truncated = safe_text[:max_chars]
        last_boundary = max(truncated.rfind("."), truncated.rfind("!"), truncated.rfind("?"))
        if last_boundary > max_chars // 2: # Ensure we don't truncate too much
            safe_text = truncated[:last_boundary + 1] + " ..."
        else:
            safe_text = truncated + " ..."

    # 2. Voice Validation
    target_voice = request.voice if request.voice and str(request.voice).startswith("en-") else "en-US-GuyNeural"
    
    logger.info(f"Generating speech ({target_voice}) for text: {safe_text[:50]}...")
    FileManager.initialize_directories()
    temp_filename = f"tts_{int(time.time())}.mp3"
    temp_path = os.path.join(FileManager.TEMP_DIR, temp_filename)
    
    try:
        # 3. Robust Engine Execution
        communicate = edge_tts.Communicate(safe_text, target_voice)
        await communicate.save(temp_path)
        
        if not os.path.exists(temp_path) or os.path.getsize(temp_path) == 0:
            raise Exception("Edge-TTS generated an empty or missing file.")
            
        return FileResponse(temp_path, media_type="audio/mpeg", filename=temp_filename)
        
    except Exception as e:
        logger.error(f"TTS Engine Failure: {str(e)}", exc_info=True)
        # Clean up failed file if it exists
        if os.path.exists(temp_path):
            try: os.remove(temp_path)
            except: pass
        raise HTTPException(status_code=500, detail="Voice engine encountered an internal error.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.server:app", host="0.0.0.0", port=8000, reload=True)
