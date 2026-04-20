import os
import json
import hashlib
import base64
import logging
from src.utils.diagnostics import get_logger, safe_execute

logger = get_logger()


def _get_fernet():
    """
    Return a Fernet instance keyed from INTERNAL_API_KEY.
    Falls back to None if cryptography is unavailable, so the rest of the
    code degrades gracefully to plain-JSON storage instead of crashing.
    """
    try:
        from cryptography.fernet import Fernet
        raw_key = os.getenv("INTERNAL_API_KEY", "")
        # Derive a 32-byte key with SHA-256 and base64-encode it for Fernet
        derived = hashlib.sha256(raw_key.encode()).digest()
        fernet_key = base64.urlsafe_b64encode(derived)
        return Fernet(fernet_key)
    except ImportError:
        logger.warning("cryptography package not installed — vault stored as plain JSON")
        return None


class FileManager:
    TEMP_DIR = "temp_data"
    LOG_DIR = "logs"
    ASSETS_DIR = "assets"

    @classmethod
    @safe_execute(log_msg="Dir Init Error")
    def initialize_directories(cls):
        """Ensures required working directories exist on startup."""
        os.makedirs(cls.TEMP_DIR, exist_ok=True)
        os.makedirs(cls.LOG_DIR, exist_ok=True)
        os.makedirs(cls.ASSETS_DIR, exist_ok=True)

    @classmethod
    @safe_execute(default_val=0, log_msg="Cleanup Error")
    def cleanup_all_data(cls):
        """Deletes all temporary audio files and logs (Privacy Feature) - Windows Safe."""
        deleted_files = 0

        # 1. Clean Audio Files
        if os.path.exists(cls.TEMP_DIR):
            for f in os.listdir(cls.TEMP_DIR):
                try:
                    full_path = os.path.join(cls.TEMP_DIR, f)
                    if os.path.isfile(full_path):
                        os.remove(full_path)
                        deleted_files += 1
                except Exception as e:
                    logger.warning(f"Could not delete temp file {f}: {e}")

        # 2. Clean Log Files
        if os.path.exists(cls.LOG_DIR):
            for f in os.listdir(cls.LOG_DIR):
                if f == "app_debug.log": continue
                try:
                    full_path = os.path.join(cls.LOG_DIR, f)
                    if os.path.isfile(full_path):
                        os.remove(full_path)
                        deleted_files += 1
                except Exception as e:
                    logger.error(f"Could not delete log {f}: {e}")

        return deleted_files

    @classmethod
    @safe_execute(log_msg="Safe Delete Error")
    def safe_delete_file(cls, file_path):
        """Safely removes a single file without crashing if it doesn't exist."""
        if file_path and os.path.exists(file_path):
            os.remove(file_path)

    # Provider label → environment variable name
    _ENV_KEY_MAP = {
        "OpenAI":        "OPENAI_API_KEY",
        "Anthropic":     "ANTHROPIC_API_KEY",
        "Google Gemini": "GOOGLE_API_KEY",
    }

    @classmethod
    @safe_execute(default_val={}, log_msg="Key Load Error")
    def load_saved_keys(cls):
        """Loads API keys: vault.json takes precedence, env vars fill any gaps."""
        keys = {}

        # 1. Seed from environment variables (lowest priority)
        for provider, env_var in cls._ENV_KEY_MAP.items():
            val = os.environ.get(env_var, "").strip()
            if val:
                keys[provider] = val

        # 2. Override with vault (highest priority)
        key_file = os.path.join(cls.TEMP_DIR, "vault.json")
        if os.path.exists(key_file):
            try:
                with open(key_file, "rb") as f:
                    raw = f.read()

                fernet = _get_fernet()
                if fernet:
                    try:
                        raw = fernet.decrypt(raw)
                    except Exception:
                        # Vault was written before encryption was enabled — re-read as plain JSON
                        logger.debug("Vault decryption failed; attempting plain-JSON fallback")
                        with open(key_file, "r") as f:
                            raw = f.read().encode()

                saved = json.loads(raw.decode())
                keys.update(saved)
            except Exception as e:
                logger.warning(f"Could not load vault.json: {e}")

        return keys

    @classmethod
    @safe_execute(log_msg="Key Save Error")
    def save_keys(cls, keys_dict):
        """Saves API keys to a local encrypted file."""
        key_file = os.path.join(cls.TEMP_DIR, "vault.json")
        payload = json.dumps(keys_dict).encode()

        fernet = _get_fernet()
        if fernet:
            payload = fernet.encrypt(payload)
            with open(key_file, "wb") as f:
                f.write(payload)
        else:
            with open(key_file, "w") as f:
                f.write(payload.decode())
