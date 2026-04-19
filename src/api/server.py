import os
import shutil
import requests
import json
import time
from dotenv import load_dotenv

# Load .env file if it exists
load_dotenv()

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends, Header, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from fastapi.responses import StreamingResponse, FileResponse
import edge_tts
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from src.utils.diagnostics import get_logger, log_system_info, safe_execute

# --- 1. ENVIRONMENT & SECURITY SETUP ---
log_system_info()
logger = get_logger()

# Strict security: No hardcoded fallback. Must be set in .env or environment.
INTERNAL_API_KEY = os.getenv("INTERNAL_API_KEY")

# Maximum allowed audio upload size: 50 MB
MAX_AUDIO_BYTES = 50 * 1024 * 1024

def verify_internal_key(x_internal_key: str = Header(None)):
    """Dependency to verify the internal API key for inter-service security."""
    if not INTERNAL_API_KEY:
        logger.critical("SECURITY ALERT: INTERNAL_API_KEY is not set in environment!")
        raise HTTPException(status_code=500, detail="Server security misconfiguration")

    if not x_internal_key or x_internal_key != INTERNAL_API_KEY:
        # Never log the attempted key value — only that an attempt was made.
        logger.warning("Unauthorized access attempt on internal API")
        raise HTTPException(status_code=401, detail="Invalid Internal API Key")
    return x_internal_key

def _sanitize_context(text: str) -> str:
    """
    Wrap user-supplied context in XML delimiters so the LLM treats it as
    data, not instructions (prompt-injection defence).
    """
    if not text:
        return ""
    # Strip any XML-like tags the user may have injected
    cleaned = text.replace("<", "&lt;").replace(">", "&gt;")
    return cleaned

# --- 2. BACKEND ENGINES ---
from src.backend.audio_processor import AudioProcessor
from src.backend.llm_client import LLMClient
from src.utils.file_manager import FileManager
from src.utils.ollama_resolver import resolve_ollama_host
from src.backend.hardware import HardwareInfo
from src.backend.monitor import ResourceMonitor

# Initialize engines at the server level
processor = AudioProcessor()
hw_info = HardwareInfo()
res_monitor = ResourceMonitor()

# --- 3. RATE LIMITER ---
limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title="Data Drifters: Interview Coach API",
    description="Backend API for AI-powered interview coaching.",
    version="1.2.0"
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# --- 4. CORS CONFIGURATION ---
# Restrict to the known Streamlit origin; extend via ALLOWED_ORIGINS env var.
_raw_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:8501,http://127.0.0.1:8501")
_allowed_origins = [o.strip() for o in _raw_origins.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-Internal-Key"],
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

class PullModelRequest(BaseModel):
    model: str

# --- ENDPOINTS ---

@app.get("/", tags=["Health"])
@app.get("/health", tags=["Health"])
def root():
    return {"status": "online", "message": "Data Drifters API is running."}

@app.get("/hardware", dependencies=[Depends(verify_internal_key)], tags=["Telemetry"])
@limiter.limit("120/minute")
def get_hardware(request: Request):
    try:
        rec_tier, rec_reason = hw_info.get_recommendation()
        stats = res_monitor.get_system_usage()

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
@limiter.limit("20/minute")
async def process_audio(
    request: Request,
    file: UploadFile = File(...),
    difficulty: str = Form("Standard Interview"),
    tier: str = Form("Balanced")
):
    FileManager.initialize_directories()

    # Enforce upload size limit before writing to disk
    content = await file.read()
    if len(content) > MAX_AUDIO_BYTES:
        raise HTTPException(status_code=413, detail=f"Audio file exceeds maximum allowed size of {MAX_AUDIO_BYTES // (1024*1024)} MB")

    # Use a random name to prevent any path-based issues from the original filename
    safe_filename = f"upload_{int(time.time() * 1000)}.audio"
    temp_path = os.path.join(FileManager.TEMP_DIR, safe_filename)
    try:
        with open(temp_path, "wb") as buffer:
            buffer.write(content)
        transcript, metrics, duration, error = processor.process_interview(
            temp_path, difficulty=difficulty, tier=tier
        )
        if error:
            raise HTTPException(status_code=500, detail="Audio processing failed")
        return {"transcript": transcript, "metrics": metrics, "duration": duration}
    except HTTPException: raise
    except Exception as e:
        logger.error(f"API Audio error: {e}")
        raise HTTPException(status_code=500, detail="Audio processing failed")
    finally:
        FileManager.safe_delete_file(temp_path)

@app.post("/test-connection", dependencies=[Depends(verify_internal_key)], tags=["LLM"])
@limiter.limit("30/minute")
def test_connection(request: Request, req: ConnectionRequest):
    try:
        llm = LLMClient(provider=req.provider, model_name=req.model,
                        compute_type=req.compute_type, api_key=req.api_key)
        success, message = llm.test_connection()
        return {"success": success, "message": message}
    except Exception as e:
        logger.error(f"Test connection error: {e}")
        raise HTTPException(status_code=500, detail="Connection test failed")

@app.post("/generate-response", dependencies=[Depends(verify_internal_key)], tags=["LLM"])
@limiter.limit("60/minute")
def generate_response(request: Request, req: LLMRequest):
    """Generates LLM response with optional resume/job context."""
    try:
        llm = LLMClient(provider=req.provider, model_name=req.model,
                        compute_type=req.compute_type, api_key=req.api_key)

        # Build contextual instruction — user content is sanitized and delimited
        context_instr = ""
        if req.resume_context:
            safe_resume = _sanitize_context(req.resume_context)
            context_instr += f"\n<resume>\n{safe_resume}\n</resume>\n"
        if req.job_context:
            safe_job = _sanitize_context(req.job_context)
            context_instr += f"\n<job_description>\n{safe_job}\n</job_description>\n"

        full_system_prompt = req.system_prompt + context_instr

        history = [msg.dict() for msg in req.chat_history] if req.chat_history else []
        response_text = llm.generate_response(system_prompt=full_system_prompt,
                                            user_message=req.user_message, chat_history=history)
        return {"response": response_text, "model_used": llm.model_name}
    except Exception as e:
        logger.error(f"Generate Response Error: {e}")
        raise HTTPException(status_code=500, detail="Response generation failed")

@app.post("/generate-response-stream", dependencies=[Depends(verify_internal_key)], tags=["LLM"])
@limiter.limit("60/minute")
def generate_response_stream(request: Request, req: LLMRequest):
    """Streams LLM response token by token for lower perceived latency."""
    try:
        llm = LLMClient(provider=req.provider, model_name=req.model,
                        compute_type=req.compute_type, api_key=req.api_key)
        context_instr = ""
        if req.resume_context:
            safe_resume = _sanitize_context(req.resume_context)
            context_instr += f"\n<resume>\n{safe_resume}\n</resume>\n"
        if req.job_context:
            safe_job = _sanitize_context(req.job_context)
            context_instr += f"\n<job_description>\n{safe_job}\n</job_description>\n"
        full_system_prompt = req.system_prompt + context_instr
        history = [msg.dict() for msg in req.chat_history] if req.chat_history else []
        return StreamingResponse(
            llm.generate_response_stream(full_system_prompt, req.user_message, history),
            media_type="text/plain"
        )
    except Exception as e:
        logger.error(f"Generate Response Stream Error: {e}")
        raise HTTPException(status_code=500, detail="Stream generation failed")

@app.post("/generate-questions", dependencies=[Depends(verify_internal_key)], tags=["LLM"])
@limiter.limit("30/minute")
def generate_questions(request: Request, req: QuestionRequest):
    """Specifically generates interview questions using full role and context metadata."""
    try:
        config = req.engine_config
        llm = LLMClient(provider=config['provider'], model_name=config['model'],
                        compute_type=config['compute'], api_key=config.get('api_key'))

        context_bonus = ""
        if req.resume_context:
            safe_resume = _sanitize_context(req.resume_context)
            context_bonus += f"\n<resume>\n{safe_resume[:2000]}\n</resume>"
        if req.job_context:
            safe_job = _sanitize_context(req.job_context)
            context_bonus += f"\n<job_description>\n{safe_job[:2000]}\n</job_description>"

        q_prompt = (
            f"Generate 3 highly specific interview questions for a {req.seniority} {req.job_title} "
            f"during the '{req.selected_round}' round in the {req.industry} industry. {context_bonus} "
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
        raise HTTPException(status_code=500, detail="Question generation failed")

@app.get("/models", dependencies=[Depends(verify_internal_key)], tags=["Local Models"])
@limiter.limit("30/minute")
def list_models(request: Request):
    candidates = []
    env_host = os.getenv("OLLAMA_HOST")
    if env_host:
        candidates.append(env_host)
    candidates.extend(["http://host.docker.internal:11434", "http://127.0.0.1:11434", "http://localhost:11434"])
    host = resolve_ollama_host(candidates)
    if not host:
        return {"models": []}
    try:
        res = requests.get(f"{host}/api/tags", timeout=2)
        if res.status_code == 200:
            return {"models": [m['name'] for m in res.json().get('models', [])]}
    except Exception as e:
        logger.debug(f"Model list fetch failed: {e}", exc_info=True)
    return {"models": []}

@app.post("/pull-model", dependencies=[Depends(verify_internal_key)], tags=["Local Models"])
@limiter.limit("10/minute")
def pull_model(request: Request, req: PullModelRequest):
    model_name = req.model
    candidates = [os.getenv("OLLAMA_HOST"), "http://host.docker.internal:11434", "http://127.0.0.1:11434"]
    ollama_host = resolve_ollama_host([h for h in candidates if h]) or "http://host.docker.internal:11434"

    def generate():
        try:
            with requests.post(f"{ollama_host}/api/pull", json={"name": model_name},
                               stream=True, timeout=None) as r:
                for chunk in r.iter_lines():
                    if chunk: yield chunk.decode('utf-8') + "\n"
        except Exception as e:
            logger.error(f"Model pull error: {e}", exc_info=True)
            yield json.dumps({"error": "Model download failed"}) + "\n"
    return StreamingResponse(generate(), media_type="application/x-ndjson")

@app.post("/generate-speech", dependencies=[Depends(verify_internal_key)], tags=["TTS"])
@limiter.limit("30/minute")
async def generate_speech(request: Request, req: SpeechRequest, background_tasks: BackgroundTasks):
    """Generates an MP3 file from text using edge-tts with safety limits."""
    max_chars = 1500
    safe_text = req.text.strip()

    if len(safe_text) > max_chars:
        logger.warning(f"TTS input too long ({len(safe_text)} chars). Truncating...")
        truncated = safe_text[:max_chars]
        last_boundary = max(truncated.rfind("."), truncated.rfind("!"), truncated.rfind("?"))
        if last_boundary > max_chars // 2:
            safe_text = truncated[:last_boundary + 1] + " ..."
        else:
            safe_text = truncated + " ..."

    target_voice = req.voice if req.voice and str(req.voice).startswith("en-") else "en-US-GuyNeural"

    logger.info(f"Generating speech ({target_voice}) for text: {safe_text[:50]}...")
    FileManager.initialize_directories()
    temp_filename = f"tts_{int(time.time())}.mp3"
    temp_path = os.path.join(FileManager.TEMP_DIR, temp_filename)

    try:
        communicate = edge_tts.Communicate(safe_text, target_voice)
        await communicate.save(temp_path)

        if not os.path.exists(temp_path) or os.path.getsize(temp_path) == 0:
            raise Exception("Edge-TTS generated an empty or missing file.")

        background_tasks.add_task(FileManager.safe_delete_file, temp_path)
        return FileResponse(temp_path, media_type="audio/mpeg", filename=temp_filename)

    except Exception as e:
        logger.error(f"TTS Engine Failure: {str(e)}", exc_info=True)
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception as cleanup_err:
                logger.debug(f"TTS temp file cleanup failed: {cleanup_err}", exc_info=True)
        raise HTTPException(status_code=500, detail="Voice engine encountered an internal error.")

if __name__ == "__main__":
    import uvicorn
    is_dev = os.getenv("ENV", "production").lower() == "development"
    uvicorn.run("src.api.server:app", host="0.0.0.0", port=8000, reload=is_dev)
