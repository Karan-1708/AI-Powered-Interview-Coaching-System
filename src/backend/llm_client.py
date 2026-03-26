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
                # Try to find the working host once at init
                self.ollama_host = "http://127.0.0.1:11434" # Default
                for host in ["http://host.docker.internal:11434", "http://127.0.0.1:11434", "http://localhost:11434"]:
                    try:
                        if requests.get(f"{host}/api/tags", timeout=0.5).status_code == 200:
                            self.ollama_host = host
                            break
                    except: continue
        else:
            self.ollama_host = None # Not needed for external
        
        logger.info(f"LLMClient initialized for {provider} (Model: {model_name})")

    @safe_execute(default_val=(False, "Connection Exception"), log_msg="LLM Test Error")
    def test_connection(self):
        """Pings the selected provider to see if it's alive."""
        if self.provider == "Local (Ollama)":
            if not self.ollama_host:
                return False, "[ERROR] Ollama Host not found. Ensure Ollama is running."
            try:
                res = requests.get(f"{self.ollama_host}/api/tags", timeout=5)
                if res.status_code == 200:
                    models = [m['name'] for m in res.json().get('models', [])]
                    # Check for exact or partial match (e.g., llama3 vs llama3:latest)
                    if any(self.model_name in m for m in models):
                        return True, f"[OK] Ollama is active. Found model: {self.model_name}"
                    return False, f"[ERROR] Ollama active, but model '{self.model_name}' not found. Download it first."
                return False, f"[ERROR] Ollama returned error: {res.status_code}"
            except Exception as e:
                return False, f"[ERROR] Ollama Connection Failed: {str(e)}"
        
        elif self.provider in ["External API (Frontier Models)", "OpenAI", "Anthropic", "Google Gemini"]:
            if not self.api_key:
                return False, "[ERROR] API Key is missing. Please enter it in the sidebar."
            
            try:
                # Use provider name or model name to determine the probe
                prov_lower = self.provider.lower()
                model_lower = self.model_name.lower()
                
                if "gemini" in prov_lower or "gemini" in model_lower:
                    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={self.api_key}"
                    res = requests.get(url, timeout=10)
                    if res.status_code == 200: return True, "[OK] Google Gemini API connection verified."
                    return False, f"[ERROR] Gemini API Error: {res.json().get('error', {}).get('message', 'Invalid Key')}"
                
                elif "openai" in prov_lower or "gpt" in model_lower:
                    # Minimal probe for OpenAI
                    url = "https://api.openai.com/v1/models"
                    headers = {"Authorization": f"Bearer {self.api_key}"}
                    res = requests.get(url, headers=headers, timeout=10)
                    if res.status_code == 200: return True, "[OK] OpenAI API connection verified."
                    return False, f"[ERROR] OpenAI API Error: {res.json().get('error', {}).get('message', 'Unauthorized')}"

                elif "anthropic" in prov_lower or "claude" in model_lower:
                    # Minimal probe for Anthropic
                    url = "https://api.anthropic.com/v1/models"
                    headers = {"x-api-key": self.api_key, "anthropic-version": "2023-06-01"}
                    res = requests.get(url, headers=headers, timeout=10)
                    if res.status_code == 200: return True, "[OK] Anthropic API connection verified."
                    return False, f"[ERROR] Anthropic API Error: {res.json().get('error', {}).get('message', 'Unauthorized')}"
                
                return True, f"[OK] Config set for {self.model_name}. Attempting connection..."
            except Exception as e:
                return False, f"[ERROR] API Probe Failed: {str(e)}"

        return False, "[ERROR] Unknown Provider Configured."

    @safe_execute(default_val="Error: LLM Generation Failed", log_msg="LLM Generation Error")
    def generate_response(self, system_prompt: str, user_message: str, chat_history: list = None):
        """Standardized generation entry point."""
        history = chat_history if chat_history else []
        
        try:
            if self.provider == "Local (Ollama)":
                return self._generate_ollama(system_prompt, user_message, history)
            
            model_lower = self.model_name.lower()
            if "gemini" in model_lower:
                return self._generate_gemini(system_prompt, user_message, history)
            elif "gpt" in model_lower:
                return self._generate_openai(system_prompt, user_message, history)
            elif "claude" in model_lower:
                return self._generate_anthropic(system_prompt, user_message, history)
            else:
                return f"Error: Model '{self.model_name}' not yet supported for direct generation."
        except Exception as e:
            logger.error(f"LLM Routing Error: {e}")
            return f"LLM Generation Failed (Routing Error): {str(e)}"

    @safe_execute(default_val="Ollama Error", log_msg="Ollama Gen Error")
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
            "options": {"temperature": 0.7, "num_predict": 500}
        }
        
        response = requests.post(url, json=payload, timeout=90)
        if response.status_code == 200:
            return response.json()["message"]["content"]
        return f"Ollama Error ({response.status_code}): {response.text}"

    @safe_execute(default_val="Gemini Error", log_msg="Gemini Gen Error")
    def _generate_gemini(self, system, user, history):
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent?key={self.api_key}"
        
        contents = []
        for msg in history:
            role = "model" if msg["role"] == "assistant" else "user"
            contents.append({"role": role, "parts": [{"text": msg["content"]}]})
        
        prompt_text = f"SYSTEM INSTRUCTION: {system}\n\nUSER MESSAGE: {user}"
        contents.append({"role": "user", "parts": [{"text": prompt_text}]})
        
        payload = {
            "contents": contents,
            "generationConfig": {"temperature": 0.7, "maxOutputTokens": 800}
        }
        
        response = requests.post(url, json=payload, timeout=60)
        if response.status_code != 200:
            err_msg = response.json().get('error', {}).get('message', 'Unknown Error')
            return f"Gemini API Error: {err_msg}"
        
        return response.json()["candidates"][0]["content"]["parts"][0]["text"]

    @safe_execute(default_val="OpenAI Error", log_msg="OpenAI Gen Error")
    def _generate_openai(self, system, user, history):
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        messages = [{"role": "system", "content": system}]
        for msg in history:
            messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": user})
        
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 800
        }
        
        response = requests.post(url, json=payload, headers=headers, timeout=60)
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        return f"OpenAI Error: {response.json().get('error', {}).get('message', 'Unknown')}"

    @safe_execute(default_val="Anthropic Error", log_msg="Anthropic Gen Error")
    def _generate_anthropic(self, system, user, history):
        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json"
        }
        
        messages = []
        for msg in history:
            messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": user})
        
        payload = {
            "model": self.model_name,
            "system": system,
            "messages": messages,
            "max_tokens": 800,
            "temperature": 0.7
        }
        
        response = requests.post(url, json=payload, headers=headers, timeout=60)
        if response.status_code == 200:
            return response.json()["content"][0]["text"]
        return f"Anthropic Error: {response.json().get('error', {}).get('message', 'Unknown')}"

