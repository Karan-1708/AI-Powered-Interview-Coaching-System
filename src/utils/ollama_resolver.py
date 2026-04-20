import requests
from src.utils.diagnostics import get_logger

logger = get_logger()


def resolve_ollama_host(candidates: list, timeout: float = 1.0) -> str | None:
    """
    Probe a list of candidate Ollama host URLs and return the first one that
    responds successfully on /api/tags.  Returns None if none are reachable.
    """
    for host in candidates:
        if not host:
            continue
        try:
            if requests.get(f"{host}/api/tags", timeout=timeout).status_code == 200:
                return host
        except Exception as e:
            logger.debug(f"Ollama host probe failed for {host}: {e}")
    return None
