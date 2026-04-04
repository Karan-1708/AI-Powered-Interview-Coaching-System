import os
import requests
import json
import logging
import re
from src.utils.diagnostics import get_logger, safe_execute

logger = get_logger()

class LLMClient:
    def __init__(self, provider: str, model_name: str, compute_type: str, api_key: str = None):
        self.provider = provider
        # Strip all strings to prevent common whitespace errors
        self.model_name = model_name.strip() if model_name else "unknown"
        self.compute_type = compute_type
        self.api_key = api_key.strip() if api_key else None
        
        # Determine Ollama Host ONLY if provider is local
        if "Local" in self.provider:
            env_host = os.getenv("OLLAMA_HOST")
            if env_host:
                self.ollama_host = env_host
            else:
                # Probing defaults only when necessary
                working_host = "http://127.0.0.1:11434"
                for host in ["http://host.docker.internal:11434", "http://127.0.0.1:11434", "http://localhost:11434"]:
                    try:
                        if requests.get(f"{host}/api/tags", timeout=1).status_code == 200:
                            working_host = host
                            break
                    except: continue
                self.ollama_host = working_host
        else:
            self.ollama_host = None
        
        logger.info(f"LLMClient initialized for {provider} using model {self.model_name}")

    @safe_execute(default_val=(False, "Connection Exception"), log_msg="LLM Test Error")
    def test_connection(self):
        """Pings the selected provider to see if it's alive."""
        logger.info(f"Testing connection for provider: '{self.provider}'")
        
        # 1. Local Ollama Case
        if "Local" in self.provider:
            try:
                res = requests.get(f"{self.ollama_host}/api/tags", timeout=5)
                if res.status_code == 200:
                    models = [m['name'] for m in res.json().get('models', [])]
                    if any(self.model_name in m for m in models):
                        return True, f"🟢 Ollama is active. Found model: {self.model_name}"
                    return False, f"🔴 Ollama active, but model '{self.model_name}' not found."
                return False, f"🔴 Ollama returned error: {res.status_code}"
            except Exception as e:
                return False, f"🔴 Ollama Connection Failed: {str(e)}"
        
        # 2. External API Case
        else:
            if not self.api_key:
                return False, "🔴 API Key is missing. Please enter it in the sidebar."
            
            p_name = self.provider.lower()
            try:
                # --- GOOGLE GEMINI ---
                if "gemini" in p_name or "google" in p_name:
                    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={self.api_key}"
                    res = requests.get(url, timeout=15)
                    if res.status_code == 200:
                        return True, f"🟢 Gemini API verified. Model '{self.model_name}' is ready."
                    
                    err_msg = res.json().get('error', {}).get('message', 'Check API Key')
                    return False, f"🔴 Gemini Error {res.status_code}: {err_msg}"
                
                # --- OPENAI ---
                elif "openai" in p_name:
                    url = "https://api.openai.com/v1/models"
                    headers = {"Authorization": f"Bearer {self.api_key}"}
                    res = requests.get(url, headers=headers, timeout=10)
                    if res.status_code == 200:
                        return True, f"🟢 OpenAI API verified. Model '{self.model_name}' is ready."
                    
                    err_msg = res.json().get('error', {}).get('message', 'Check API Key')
                    return False, f"🔴 OpenAI Error {res.status_code}: {err_msg}"

                # --- ANTHROPIC ---
                elif "anthropic" in p_name:
                    if not self.api_key.startswith("sk-ant-"):
                        return False, "🔴 Anthropic Error: Key must start with 'sk-ant-'. Please check your key."

                    url = "https://api.anthropic.com/v1/messages"
                    headers = {
                        "x-api-key": self.api_key,
                        "anthropic-version": "2023-06-01",
                        "content-type": "application/json"
                    }
                    
                    # Use a known-good model for the connection test if the selected one is custom/broken
                    probe_model = self.model_name if "claude" in self.model_name else "claude-3-5-sonnet-20241022"
                    
                    payload = {
                        "model": probe_model,
                        "max_tokens": 1,
                        "messages": [{"role": "user", "content": "Hi"}]
                    }
                    res = requests.post(url, headers=headers, json=payload, timeout=10)
                    
                    if res.status_code == 200:
                        return True, f"🟢 Anthropic API verified. Model '{self.model_name}' is ready."
                    
                    # Enhanced 404/400 Diagnostic
                    try:
                        err_data = res.json()
                        err_msg = err_data.get('error', {}).get('message', res.text)
                    except:
                        err_msg = res.text
                    
                    if res.status_code == 404:
                        return False, f"🔴 Anthropic Error 404: The model '{probe_model}' was not found. This key might not have access to it yet."
                        
                    return False, f"🔴 Anthropic Error {res.status_code}: {err_msg}"
                
                return True, f"🟢 {self.provider} config set for {self.model_name}."
            except Exception as e:
                return False, f"🔴 API Probe Failed: {str(e)}"

    @safe_execute(default_val="Error: LLM Generation Failed", log_msg="LLM Generation Error")
    def generate_response(self, system_prompt: str, user_message: str, chat_history: list = None):
        """Standardized generation entry point."""
        history = chat_history if chat_history else []
        p_name = self.provider.lower()
        
        if "Local" in self.provider:
            return self._generate_ollama(system_prompt, user_message, history)
        elif "gemini" in p_name or "google" in p_name:
            return self._generate_gemini(system_prompt, user_message, history)
        elif "openai" in p_name:
            return self._generate_openai(system_prompt, user_message, history)
        elif "anthropic" in p_name:
            return self._generate_anthropic(system_prompt, user_message, history)
        else:
            return f"Error: Provider '{self.provider}' is not supported for direct generation."

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
        
        full_user_text = f"SYSTEM INSTRUCTION: {system}\n\nUSER MESSAGE: {user}"
        contents.append({"role": "user", "parts": [{"text": full_user_text}]})
        
        payload = {
            "contents": contents,
            "generationConfig": {"temperature": 0.7, "maxOutputTokens": 1000}
        }
        
        response = requests.post(url, json=payload, timeout=30)
        if response.status_code != 200:
            try:
                err = response.json().get('error', {}).get('message', 'Unknown Error')
            except:
                err = response.text
            return f"Gemini API Error {response.status_code}: {err}"
        
        try:
            return response.json()["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError):
            return "Gemini Error: Unexpected response format from API."

    def _generate_openai(self, system, user, history):
        url = "https://api.openai.com/v1/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        
        # Reasoning models (o1, o3, o4, gpt-5, etc.) use 'max_completion_tokens'
        # and do not support the 'system' role or 'temperature' adjustments in the same way.
        is_reasoning_model = bool(re.match(r"^(o\d|gpt-5)", self.model_name.lower()))
        
        messages = []
        if is_reasoning_model:
            messages.append({"role": "user", "content": f"INSTRUCTIONS: {system}"})
        else:
            messages.append({"role": "system", "content": system})
            
        for msg in history:
            messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": user})
        
        payload = {
            "model": self.model_name,
            "messages": messages
        }
        
        if is_reasoning_model:
            payload["max_completion_tokens"] = 1000
            # Temperature is not supported or must be 1.0 for many reasoning models
        else:
            payload["max_tokens"] = 800
            payload["temperature"] = 0.7
        
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        
        try:
            err = response.json().get('error', {}).get('message', 'Unknown error')
        except:
            err = response.text
        return f"OpenAI Error {response.status_code}: {err}"

    def _generate_anthropic(self, system, user, history):
        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
        
        messages = []
        for msg in history:
            messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": user})
        
        payload = {
            "model": self.model_name,
            "system": system,
            "messages": messages,
            "max_tokens": 1000,
            "temperature": 0.7
        }
        
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        if response.status_code == 200:
            return response.json()["content"][0]["text"]
        
        try:
            err = response.json().get('error', {}).get('message', 'Unknown error')
        except:
            err = response.text
        
        if response.status_code == 404:
            return f"Anthropic Error 404: The model '{self.model_name}' was not found. Please check model availability for your account/region."
            
        return f"Anthropic Error {response.status_code}: {err}"
