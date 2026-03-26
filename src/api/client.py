import os
import requests
import json
import logging
from src.utils.diagnostics import get_logger, safe_execute

logger = get_logger()

class APIClient:
    """The bridge between the Streamlit Frontend and the FastAPI Backend."""
    
    BASE_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
    # Pull internal key for authentication with the backend
    INTERNAL_KEY = os.getenv("INTERNAL_API_KEY", "dev-key-12345")

    @staticmethod
    def _get_headers():
        """Returns standard headers including the internal security key."""
        return {
            "X-Internal-Key": APIClient.INTERNAL_KEY
        }

    @staticmethod
    @safe_execute(default_val=None, log_msg="Telemetry Fetch Error")
    def get_hardware_status():
        """Fetches live telemetry from the backend API container."""
        url = f"{APIClient.BASE_URL}/hardware"
        res = requests.get(url, headers=APIClient._get_headers(), timeout=5)
        if res.status_code == 200:
            return res.json()
        return None

    @staticmethod
    @safe_execute(default_val=(None, None, None, "API Error"), log_msg="Audio Upload Error")
    def process_audio(file_path, difficulty, compute_mode):
        """Sends the audio file to the backend API for transcription and analysis."""
        url = f"{APIClient.BASE_URL}/process-audio"
        with open(file_path, "rb") as f:
            files = {"file": (os.path.basename(file_path), f, "audio/wav")}
            data = {"difficulty": difficulty, "compute_mode": compute_mode}
            response = requests.post(
                url, 
                files=files, 
                data=data, 
                headers=APIClient._get_headers(), 
                timeout=120 # Increased for simultaneous users
            )
        
        if response.status_code == 200:
            res = response.json()
            return res['transcript'], res['metrics'], res['duration'], None
        return None, None, None, f"API {response.status_code}: {response.text}"

    @staticmethod
    @safe_execute(default_val="Generation Error", log_msg="LLM Generation Error")
    def generate_response(system_prompt, user_message, chat_history, engine_config):
        """Generates a text response using the selected LLM provider."""
        url = f"{APIClient.BASE_URL}/generate-response"
        payload = {
            "system_prompt": system_prompt,
            "user_message": user_message,
            "chat_history": chat_history,
            "provider": engine_config['provider'],
            "model": engine_config['model'],
            "compute_type": engine_config['compute'],
            "api_key": engine_config['api_key']
        }
        response = requests.post(
            url, 
            json=payload, 
            headers=APIClient._get_headers(), 
            timeout=60 # Increased for external APIs
        )
        if response.status_code == 200:
            return response.json()['response']
        return f"API Error: {response.text}"

    @staticmethod
    @safe_execute(default_val=(False, "Connection Failure"), log_msg="Connection Test Error")
    def test_connection(engine_config):
        """Verifies connection to the LLM provider."""
        url = f"{APIClient.BASE_URL}/test-connection"
        payload = {
            "provider": engine_config['provider'],
            "model": engine_config['model'],
            "compute_type": engine_config['compute'],
            "api_key": engine_config['api_key']
        }
        res = requests.post(
            url, 
            json=payload, 
            headers=APIClient._get_headers(), 
            timeout=30 # Increased for cold starts
        )
        if res.status_code == 200:
            data = res.json()
            return data['success'], data['message']
        return False, f"🔴 API Protocol Error: {res.text}"

    @staticmethod
    @safe_execute(default_val=["llama3.1", "gemma2"], log_msg="Model Fetch Error")
    def get_local_models():
        """Fetches models available in the local Ollama instance."""
        url = f"{APIClient.BASE_URL}/models"
        res = requests.get(url, headers=APIClient._get_headers(), timeout=5)
        if res.status_code == 200:
            models = res.json().get('models', [])
            return models if models else ["llama3.1"]
        return ["llama3.1"]

    @staticmethod
    def pull_model_stream(model_name):
        """
        Streams progress of downloading a new model.
        Timeout set to None because model pulls can have long idle periods.
        """
        url = f"{APIClient.BASE_URL}/pull-model"
        return requests.post(
            url, 
            json={"model": model_name}, 
            headers=APIClient._get_headers(), 
            stream=True, 
            timeout=None 
        )
