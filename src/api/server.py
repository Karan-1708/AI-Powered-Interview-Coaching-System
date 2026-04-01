import os
import shutil
import requests
import json
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from fastapi.responses import StreamingResponse
from src.utils.diagnostics import get_logger, log_system_info, safe_execute

# --- 1. ENVIRONMENT & SECURITY SETUP ---
log_system_info()
logger = get_logger()

# This must match what you set in your Streamlit Cloud "Secrets"
INTERNAL_API_KEY = os.getenv("INTERNAL_API_KEY", "dev-key-12345")

def verify_internal_key(x_internal_key: str = Header(None)):
    """Dependency to verify the internal API key for inter-service security."""
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
    version="1.1.0"
)

# --- 3. CORS CONFIGURATION (Required for Streamlit Cloud) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows Streamlit Cloud to connect
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

class ConnectionRequest(BaseModel):
    provider: str
    model: str
    compute_type: str
    api_key: Optional[str] = None

# --- ENDPOINTS ---

@app.get("/", tags=["Health"])
def root():
    """Simple health check."""
    return {"status": "online", "message": "Data Drifters API is running."}

@app.get("/hardware", dependencies=[Depends(verify_internal_key)], tags=["Telemetry"])
def get_hardware():
    """Returns live hardware stats. 401 if key invalid, 500 if collection fails."""
    try:
        rec_tier, rec_reason = hw_info.get_recommendation()
        stats = res_monitor.get_system_usage()
        return {
            "tier": rec_tier,
            "reason": rec_reason,
            "has_nvidia": hw_info.has_nvidia,
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
    """Transcribes audio and performs scoring. Uses threadpool to prevent blocking."""
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
    """Verifies connection to LLM provider. 401 if key invalid."""
    try:
        llm = LLMClient(provider=request.provider, model_name=request.model, 
                        compute_type=request.compute_type, api_key=request.api_key)
        success, message = llm.test_connection()
        return {"success": success, "message": message}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-response", dependencies=[Depends(verify_internal_key)], tags=["LLM"])
def generate_response(request: LLMRequest):
    """Generates LLM response. Uses threadpool to prevent UI freezing."""
    try:
        llm = LLMClient(provider=request.provider, model_name=request.model, 
                        compute_type=request.compute_type, api_key=request.api_key)
        history = [msg.dict() for msg in request.chat_history] if request.chat_history else []
        response_text = llm.generate_response(system_prompt=request.system_prompt, 
                                            user_message=request.user_message, chat_history=history)
        return {"response": response_text, "model_used": llm.model_name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models", dependencies=[Depends(verify_internal_key)], tags=["Local Models"])
def list_models():
    """Fetches local Ollama models."""
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
    """Streams model download from Ollama."""
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.server:app", host="0.0.0.0", port=8000, reload=True)
