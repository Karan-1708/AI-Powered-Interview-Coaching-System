import os
import requests
import json
from src.utils.diagnostics import get_logger, safe_execute

logger = get_logger()

class LLMClient:
    def __init__(self, provider: str, model_name: str, compute_type: str, api_key: str = None):
        self.provider = provider
        self.model_name = model_name
        self.compute_type = compute_type
        self.api_key = api_key.strip() if api_key else None
        
        # --- ROBUST OLLAMA HOST DETECTION ---
        if self.provider == "Local (Ollama)":
            env_host = os.getenv("OLLAMA_HOST")
            if env_host:
                self.ollama_host = env_host
            else:
                working_host = "http://127.0.0.1:11434" # Safest default
                for host in ["http://host.docker.internal:11434", "http://127.0.0.1:11434", "http://localhost:11434"]:
                    try:
                        if requests.get(f"{host}/api/tags", timeout=1).status_code == 200:
                            working_host = host
                            break
                    except: continue
                self.ollama_host = working_host
        else:
            self.ollama_host = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
        
        logger.info(f"LLMClient initialized for {provider} using {self.ollama_host}")

    @safe_execute(default_val=(False, "Connection Exception"), log_msg="LLM Test Error")
    def test_connection(self):
        """Pings the selected provider to see if it's alive."""
        if self.provider == "Local (Ollama)":
            try:
                res = requests.get(f"{self.ollama_host}/api/tags", timeout=5)
                if res.status_code == 200:
                    models = [m['name'] for m in res.json().get('models', [])]
                    if any(self.model_name in m for m in models):
                        return True, f"🟢 Ollama is active. Found model: {self.model_name}"
                    return False, f"🔴 Ollama active, but model '{self.model_name}' not found. Download it first."
                return False, f"🔴 Ollama returned error: {res.status_code}"
            except Exception as e:
                return False, f"🔴 Ollama Connection Failed: {str(e)}"
        
        elif self.provider == "External API (Frontier Models)":
            if not self.api_key:
                return False, "🔴 API Key is missing. Please enter it in the sidebar."
            
            # Simple probe based on service
            try:
                if "gemini" in self.model_name.lower():
                    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={self.api_key}"
                    res = requests.get(url, timeout=5)
                    if res.status_code == 200: return True, "🟢 Google Gemini API connection verified."
                    return False, f"🔴 Gemini API Error: {res.json().get('error', {}).get('message', 'Invalid Key')}"
                
                # Default success for others as deep probing is expensive
                return True, f"🟢 Config set for {self.model_name}. Attempting connection..."
            except Exception as e:
                return False, f"🔴 API Probe Failed: {str(e)}"

        return False, "🔴 Unknown Provider Configured."

    @safe_execute(default_val="Error: LLM Generation Failed", log_msg="LLM Generation Error")
    def generate_response(self, system_prompt: str, user_message: str, chat_history: list = None):
        """Standardized generation entry point."""
        history = chat_history if chat_history else []
        
        if self.provider == "Local (Ollama)":
            return self._generate_ollama(system_prompt, user_message, history)
        elif "gemini" in self.model_name.lower():
            return self._generate_gemini(system_prompt, user_message, history)
        else:
            return "Error: Provider or Model not yet supported for direct generation."

    def _generate_ollama(self, system, user, history):
        url = f"{self.ollama_host}/api/chat"
        messages = [{"role": "system", "content": system}]
        for msg in history:
            messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": user})
        
        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "options": {"temperature": 0.7, "num_predict": 300}
        }
        
        response = requests.post(url, json=payload, timeout=60)
        if response.status_code == 200:
            return response.json()["message"]["content"]
        return f"Ollama Error: {response.text}"

    def _generate_gemini(self, system, user, history):
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent?key={self.api_key}"
        
        contents = []
        for msg in history:
            role = "model" if msg["role"] == "assistant" else "user"
            contents.append({"role": role, "parts": [{"text": msg["content"]}]})
        
        # Add system instruction (Gemini 1.5 style)
        payload = {
            "contents": contents + [{"role": "user", "parts": [{"text": f"SYSTEM INSTRUCTION: {system}\n\nUSER MESSAGE: {user}"}]}],
            "generationConfig": {"temperature": 0.7, "maxOutputTokens": 500}
        }
        
        response = requests.post(url, json=payload, timeout=30)
        if response.status_code != 200:
            return f"Gemini API Error: {response.json().get('error', {}).get('message', 'Unknown Error')}"
        
        try:
            return response.json()["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError):
            return "Gemini Error: Unexpected response format from API."
