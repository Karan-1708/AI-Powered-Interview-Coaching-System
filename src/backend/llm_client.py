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
        if "Ollama" in self.provider:
            try:
                res = requests.get(f"{self.ollama_host}/api/tags", timeout=3)
                if res.status_code == 200:
                    models = [m['name'] for m in res.json().get('models', [])]
                    if any(self.model_name in m for m in models):
                        return True, f"🟢 Ollama is active. Found model: {self.model_name}"
                    return False, f"🔴 Ollama is running, but model '{self.model_name}' is not found. Please download it using the 'Download Model' tool below."
                return False, "🔴 Ollama API is not responding correctly. It might be initializing."
            except Exception:
                return False, "🔴 Ollama is not running or not installed. Please start the Ollama application on your system."
        
        # 2. External API Case
        else:
            if not self.api_key:
                return False, "🔴 API Key is missing. Please enter it in the configuration sidebar."
            
            p_name = self.provider.lower()
            
            # --- FORMAT VALIDATION ---
            if "openai" in p_name and not self.api_key.startswith(("sk-proj-", "sk-")):
                return False, "🔴 Invalid OpenAI Key Format: Keys should typically start with 'sk-proj-' or 'sk-'."
            
            if "anthropic" in p_name and not self.api_key.startswith("sk-ant-"):
                return False, "🔴 Invalid Anthropic Key Format: Keys must start with 'sk-ant-'."

            try:
                # --- GOOGLE GEMINI ---
                if "gemini" in p_name or "google" in p_name:
                    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={self.api_key}"
                    res = requests.get(url, timeout=10)
                    if res.status_code == 200:
                        return True, f"🟢 Gemini API verified."
                    return False, "🔴 Gemini API Key is incorrect or invalid. Please check your Google AI Studio dashboard."
                
                # --- OPENAI ---
                elif "openai" in p_name:
                    url = "https://api.openai.com/v1/models"
                    headers = {"Authorization": f"Bearer {self.api_key}"}
                    res = requests.get(url, headers=headers, timeout=10)
                    if res.status_code == 200:
                        return True, f"🟢 OpenAI API verified."
                    return False, "🔴 OpenAI API Key is incorrect or invalid. Please check your OpenAI platform dashboard."

                # --- ANTHROPIC ---
                elif "anthropic" in p_name:
                    url = "https://api.anthropic.com/v1/messages"
                    headers = {
                        "x-api-key": self.api_key,
                        "anthropic-version": "2023-06-01",
                        "content-type": "application/json"
                    }
                    # Use a small, universally accessible model to validate the key only.
                    # The user's selected model is tested on first actual generation.
                    payload = {
                        "model": "claude-3-haiku-20240307",
                        "max_tokens": 1,
                        "messages": [{"role": "user", "content": "Hi"}]
                    }
                    res = requests.post(url, headers=headers, json=payload, timeout=10)
                    if res.status_code == 200:
                        return True, f"🟢 Anthropic API verified."
                    elif res.status_code == 401:
                        return False, "🔴 Anthropic API Key is incorrect or expired."
                    else:
                        try:
                            err = res.json().get("error", {}).get("message", res.text)
                        except Exception:
                            err = res.text
                        return False, f"🔴 Anthropic Error {res.status_code}: {err}"
                
                return True, f"🟢 {self.provider} configured."
            except Exception as e:
                return False, f"🔴 API Connection Failed: {str(e)}"

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
            "options": {"temperature": 0.7, "num_predict": 1000}
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
            payload["max_completion_tokens"] = 8000 # Increased for full feedback
            # Temperature is not supported or must be 1.0 for many reasoning models
        else:
            payload["max_tokens"] = 6000
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
            "max_tokens": 4000,
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

    # --- STREAMING METHODS ---

    def generate_response_stream(self, system_prompt: str, user_message: str, chat_history: list = None):
        """Streaming entry point — yields text chunks as they arrive from the provider."""
        history = chat_history if chat_history else []
        p_name = self.provider.lower()
        try:
            if "Local" in self.provider:
                yield from self._stream_ollama(system_prompt, user_message, history)
            elif "gemini" in p_name or "google" in p_name:
                yield from self._stream_gemini(system_prompt, user_message, history)
            elif "openai" in p_name:
                yield from self._stream_openai(system_prompt, user_message, history)
            elif "anthropic" in p_name:
                yield from self._stream_anthropic(system_prompt, user_message, history)
            else:
                yield f"Error: Provider '{self.provider}' does not support streaming."
        except Exception as e:
            logger.error(f"Streaming Error: {e}")
            yield f"[Streaming error: {str(e)}]"

    def _stream_ollama(self, system, user, history):
        url = f"{self.ollama_host}/api/chat"
        messages = [{"role": "system", "content": system}]
        for msg in history:
            messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": user})
        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": True,
            "options": {"temperature": 0.7, "num_predict": 1000}
        }
        with requests.post(url, json=payload, stream=True, timeout=60) as response:
            for line in response.iter_lines():
                if line:
                    data = json.loads(line)
                    if not data.get("done", False):
                        yield data.get("message", {}).get("content", "")

    def _stream_gemini(self, system, user, history):
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:streamGenerateContent?alt=sse&key={self.api_key}"
        contents = []
        for msg in history:
            role = "model" if msg["role"] == "assistant" else "user"
            contents.append({"role": role, "parts": [{"text": msg["content"]}]})
        contents.append({"role": "user", "parts": [{"text": f"SYSTEM INSTRUCTION: {system}\n\nUSER MESSAGE: {user}"}]})
        payload = {"contents": contents, "generationConfig": {"temperature": 0.7, "maxOutputTokens": 1000}}
        with requests.post(url, json=payload, stream=True, timeout=30) as response:
            for line in response.iter_lines():
                if line:
                    line = line.decode("utf-8")
                    if line.startswith("data:"):
                        try:
                            data = json.loads(line[5:].strip())
                            text = data["candidates"][0]["content"]["parts"][0]["text"]
                            if text:
                                yield text
                        except (KeyError, IndexError, json.JSONDecodeError):
                            continue

    def _stream_openai(self, system, user, history):
        url = "https://api.openai.com/v1/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        is_reasoning_model = bool(re.match(r"^(o\d|gpt-5)", self.model_name.lower()))
        messages = []
        if is_reasoning_model:
            messages.append({"role": "user", "content": f"INSTRUCTIONS: {system}"})
        else:
            messages.append({"role": "system", "content": system})
        for msg in history:
            messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": user})
        payload = {"model": self.model_name, "messages": messages, "stream": True}
        if is_reasoning_model:
            payload["max_completion_tokens"] = 8000
        else:
            payload["max_tokens"] = 6000
            payload["temperature"] = 0.7
        with requests.post(url, headers=headers, json=payload, stream=True, timeout=30) as response:
            for line in response.iter_lines():
                if line:
                    line = line.decode("utf-8")
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str == "[DONE]":
                            break
                        try:
                            delta = json.loads(data_str)["choices"][0]["delta"].get("content", "")
                            if delta:
                                yield delta
                        except (KeyError, IndexError, json.JSONDecodeError):
                            continue

    def _stream_anthropic(self, system, user, history):
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
            "max_tokens": 4000,
            "temperature": 0.7,
            "stream": True
        }
        with requests.post(url, headers=headers, json=payload, stream=True, timeout=30) as response:
            for line in response.iter_lines():
                if line:
                    line = line.decode("utf-8")
                    if line.startswith("data:"):
                        try:
                            data = json.loads(line[5:].strip())
                            if data.get("type") == "content_block_delta":
                                yield data["delta"].get("text", "")
                        except (KeyError, json.JSONDecodeError):
                            continue
