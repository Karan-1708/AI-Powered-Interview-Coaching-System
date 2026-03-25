import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import os
import io

# Set the internal API key for testing
INTERNAL_KEY = "test-key-12345"
os.environ["INTERNAL_API_KEY"] = INTERNAL_KEY

# Import the app after setting the environment variable
from src.api.server import app

client = TestClient(app)
HEADERS = {"X-Internal-Key": INTERNAL_KEY}

def test_root():
    """Test the health check endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "online", "message": "Data Drifters API is running."}

@patch("src.api.server.LLMClient")
def test_test_connection(mock_llm_class):
    """Test the /test-connection endpoint with mocked LLMClient."""
    mock_llm_instance = mock_llm_class.return_value
    mock_llm_instance.test_connection.return_value = (True, "Connection Successful")
    
    payload = {
        "provider": "Local (Ollama)",
        "model": "llama3.1",
        "compute_type": "NVIDIA GPU",
        "api_key": None
    }
    
    response = client.post("/test-connection", json=payload, headers=HEADERS)
    assert response.status_code == 200
    assert response.json() == {"success": True, "message": "Connection Successful"}
    mock_llm_instance.test_connection.assert_called_once()

@patch("src.api.server.LLMClient")
def test_generate_response(mock_llm_class):
    """Test the /generate-response endpoint with mocked LLMClient."""
    mock_llm_instance = mock_llm_class.return_value
    mock_llm_instance.generate_response.return_value = "This is a mock response"
    mock_llm_instance.model_name = "llama3.1"
    
    payload = {
        "system_prompt": "You are a coach",
        "user_message": "Hello",
        "chat_history": [],
        "provider": "Local (Ollama)",
        "model": "llama3.1",
        "compute_type": "NVIDIA GPU"
    }
    
    response = client.post("/generate-response", json=payload, headers=HEADERS)
    assert response.status_code == 200
    assert response.json() == {"response": "This is a mock response", "model_used": "llama3.1"}
    mock_llm_instance.generate_response.assert_called_once()

@patch("src.api.server.processor")
def test_process_audio(mock_processor):
    """Test the /process-audio endpoint with mocked AudioProcessor."""
    # Mock the return value of process_interview
    # transcript, metrics, duration, error
    mock_processor.process_interview.return_value = (
        "Hello this is a test transcript",
        {"wpm": 150, "filler_count": 2},
        5.5,
        None
    )
    
    # Create a dummy wav file
    audio_content = b"fake audio data"
    audio_file = io.BytesIO(audio_content)
    
    files = {"file": ("test.wav", audio_file, "audio/wav")}
    data = {"difficulty": "Standard Interview", "tier": "Balanced"}
    
    response = client.post("/process-audio", files=files, data=data, headers=HEADERS)
    
    assert response.status_code == 200
    res_json = response.json()
    assert res_json["transcript"] == "Hello this is a test transcript"
    assert res_json["metrics"] == {"wpm": 150, "filler_count": 2}
    assert res_json["duration"] == 5.5
    mock_processor.process_interview.assert_called_once()

def test_unauthorized_access():
    """Test that requests without the internal key are rejected."""
    response = client.get("/hardware")
    assert response.status_code == 403
    assert response.json() == {"detail": "Invalid Internal API Key"}
