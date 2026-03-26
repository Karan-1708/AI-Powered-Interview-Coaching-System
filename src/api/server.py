import os
import shutil
import requests
import json
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends, Header
from pydantic import BaseModel
from typing import List, Optional
from fastapi.responses import StreamingResponse
from src.utils.diagnostics import get_logger, log_system_info, safe_execute

# --- 1. ENVIRONMENT & SECURITY SETUP ---
log_system_info()
logger = get_logger()

INTERNAL_API_KEY = os.getenv("INTERNAL_API_KEY", "dev-key-12345")

def verify_internal_key(x_internal_key: str = Header(None)):
    """
    Dependency to verify the internal API key for inter-service security.
    
    Args:
        x_internal_key (str): The 'X-Internal-Key' provided in the request headers.
        
    Raises:
        HTTPException: 401 Unauthorized if the key is missing or invalid.
    """
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
    description="Backend API for AI-powered interview coaching, providing audio transcription, acoustic analysis, and LLM responses.",
    version="1.0.0"
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

# --- 0. HEALTH CHECK ---
@app.get("/", tags=["Health"])
def root():
    """
    Simple health check to verify the API is online.
    
    Returns:
        dict: A status message indicating the API is running.
    """
    return {"status": "online", "message": "Data Drifters API is running."}

# --- HARDWARE TELEMETRY ---
@app.get("/hardware", dependencies=[Depends(verify_internal_key)], tags=["Telemetry"])
def get_hardware():
    """
    Returns the API container's live hardware stats and GPU status.
    
    Returns:
        dict: Hardware tier recommendation, GPU availability, and detailed usage stats (CPU, RAM, VRAM).
        
    Errors:
        - 401: Invalid Internal API Key.
        - 500: Internal server error during telemetry collection.
    """
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

# --- 1. AUDIO PROCESSING ---
@app.post("/process-audio", dependencies=[Depends(verify_internal_key)], tags=["AI Processing"])
def process_audio(
    file: UploadFile = File(...), 
    difficulty: str = Form("Standard Interview"), 
    compute_mode: str = Form("CPU & RAM Core")
):
    """
    Transcribes an uploaded audio file and performs acoustic scoring.
    Uses FastAPI threadpool to avoid blocking the event loop.
    
    Args:
        file (UploadFile): The .wav or .mp3 audio file to process.
        difficulty (str): The interview difficulty tier for scoring thresholds.
        compute_mode (str): The hardware target (NVIDIA GPU or CPU & RAM Core).
        
    Returns:
        dict: Transcription text, acoustic metrics (WPM, fillers, etc.), and processing duration.
        
    Errors:
        - 401: Invalid Internal API Key.
        - 500: Failed to load Whisper model or error during transcription/scoring.
    """
    FileManager.initialize_directories()
    temp_path = os.path.join(FileManager.TEMP_DIR, f"upload_{file.filename}")
    
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        transcript, metrics, duration, error = processor.process_interview(
            temp_path, difficulty=difficulty, compute_mode=compute_mode
        )
        
        if error:
            logger.error(f"Audio processing error: {error}")
            raise HTTPException(status_code=500, detail=error)
            
        return {
            "transcript": transcript,
            "metrics": metrics,
            "duration": duration
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"API Audio processing exception: {e}")
        raise HTTPException(status_code=500, detail=f"Audio processing failed: {str(e)}")
    finally:
        FileManager.safe_delete_file(temp_path)

# --- 2. CONNECTION TEST ---
@app.post("/test-connection", dependencies=[Depends(verify_internal_key)], tags=["LLM"])
def test_connection(request: ConnectionRequest):
    """
    Verifies the connection to the specified LLM provider (Ollama or External API).
    
    Args:
        request (ConnectionRequest): Provider, model, and API key details.
        
    Returns:
        dict: Success status and a descriptive status message.
        
    Errors:
        - 401: Invalid Internal API Key.
        - 500: API or network error during the connection test.
    """
    try:
        llm = LLMClient(
            provider=request.provider,
            model_name=request.model,
            compute_type=request.compute_type,
            api_key=request.api_key
        )
        success, message = llm.test_connection()
        return {"success": success, "message": message}
    except Exception as e:
        logger.error(f"API Connection test exception: {e}")
        raise HTTPException(status_code=500, detail=f"Connection test failed: {str(e)}")

# --- 3. LLM GENERATION ---
@app.post("/generate-response", dependencies=[Depends(verify_internal_key)], tags=["LLM"])
def generate_response(request: LLMRequest):
    """
    Generates a response from an LLM based on system prompts and user messages.
    
    Args:
        request (LLMRequest): Complete context including prompts, history, and model configuration.
        
    Returns:
        dict: Generated response text and the name of the model used.
        
    Errors:
        - 401: Invalid Internal API Key.
        - 500: LLM provider error or timeout.
    """
    try:
        llm = LLMClient(
            provider=request.provider,
            model_name=request.model,
            compute_type=request.compute_type,
            api_key=request.api_key
        )
        
        history = [msg.dict() for msg in request.chat_history] if request.chat_history else []
        
        response_text = llm.generate_response(
            system_prompt=request.system_prompt,
            user_message=request.user_message,
            chat_history=history
        )
        
        return {"response": response_text, "model_used": llm.model_name}
        
    except Exception as e:
        logger.error(f"API LLM generation exception: {e}")
        raise HTTPException(status_code=500, detail=f"LLM generation failed: {str(e)}")

# --- 4. OLLAMA MODELS ---
@app.get("/models", dependencies=[Depends(verify_internal_key)], tags=["Local Models"])
def list_models():
    """
    Fetches the list of models already downloaded in the local Ollama instance.
    
    Returns:
        dict: List of available model tags.
        
    Errors:
        - 401: Invalid Internal API Key.
    """
    ollama_hosts = []
    env_host = os.getenv("OLLAMA_HOST")
    if env_host: ollama_hosts.append(env_host)
    ollama_hosts.extend(["http://host.docker.internal:11434", "http://127.0.0.1:11434", "http://localhost:11434"])
    
    for host in ollama_hosts:
        try:
            res = requests.get(f"{host}/api/tags", timeout=2)
            if res.status_code == 200:
                models = [m['name'] for m in res.json().get('models', [])]
                return {"models": models}
        except: continue
    return {"models": []}

@app.post("/pull-model", dependencies=[Depends(verify_internal_key)], tags=["Local Models"])
def pull_model(request: dict):
    """
    Downloads a new model from the Ollama library.
    Returns a streaming response.
    
    Args:
        request (dict): Contains the "model" tag to pull.
        
    Returns:
        StreamingResponse: A text stream of the download progress from Ollama.
        
    Errors:
        - 401: Invalid Internal API Key.
        - 500: Connection error to the local Ollama service.
    """
    model_name = request.get("model")
    ollama_host = "http://host.docker.internal:11434"
    for host in [os.getenv("OLLAMA_HOST"), "http://host.docker.internal:11434", "http://127.0.0.1:11434"]:
        if not host: continue
        try:
            if requests.get(f"{host}/api/tags", timeout=1).status_code == 200:
                ollama_host = host
                break
        except: continue

    def generate():
        try:
            # Note: stream=True with None timeout for the backend-to-ollama connection
            with requests.post(f"{ollama_host}/api/pull", 
                               json={"name": model_name}, stream=True, timeout=None) as r:
                for chunk in r.iter_lines():
                    if chunk:
                        yield chunk.decode('utf-8') + "\n"
        except Exception as e:
            logger.error(f"Pull stream error: {e}")
            yield json.dumps({"error": str(e)}) + "\n"

    return StreamingResponse(generate(), media_type="application/x-ndjson")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.server:app", host="0.0.0.0", port=8000, reload=True)
