"""OpenRouter chat client (OpenAI-compatible) with SSE streaming."""

import json
import time
from collections.abc import Iterator

import httpx

_RETRIES = 4
_BACKOFF = 5  # seconds, doubled per retry


def _parse_completion(data: dict) -> str:
    if "error" in data:
        raise OpenRouterError(str(data["error"].get("message", data["error"]))[:300])
    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as exc:
        raise OpenRouterError(f"unexpected response shape: {str(data)[:300]}") from exc


class OpenRouterError(RuntimeError):
    pass

from rag.config import LLMConfig, Secrets

BASE_URL = "https://openrouter.ai/api/v1/chat/completions"


class OpenRouterLLM:
    def __init__(self, cfg: LLMConfig, secrets: Secrets):
        if not secrets.openrouter_api_key:
            raise RuntimeError("RAG_OPENROUTER_API_KEY is not set in .env")
        self.cfg = cfg
        self.headers = {
            "Authorization": f"Bearer {secrets.openrouter_api_key}",
            "HTTP-Referer": "https://github.com/prod-rag",
            "X-Title": "prod-rag",
        }

    def complete(self, messages: list[dict], model: str | None = None,
                 max_tokens: int | None = None) -> str:
        payload = {
            "model": model or self.cfg.answer_model,
            "messages": messages,
            "temperature": self.cfg.temperature,
            "max_tokens": max_tokens or self.cfg.max_tokens,
        }
        last_error: Exception | None = None
        with httpx.Client(timeout=120) as client:
            for attempt in range(_RETRIES):
                resp = client.post(BASE_URL, headers=self.headers, json=payload)
                if resp.status_code == 429:  # rate limited (free tier especially)
                    last_error = OpenRouterError("rate limited (429)")
                    time.sleep(_BACKOFF * (2 ** attempt))
                    continue
                resp.raise_for_status()
                try:
                    return _parse_completion(resp.json())
                except OpenRouterError as exc:
                    last_error = exc
                    if "rate" not in str(exc).lower():
                        raise
                    time.sleep(_BACKOFF * (2 ** attempt))
        raise last_error or OpenRouterError("retries exhausted")

    def stream(self, messages: list[dict], model: str | None = None) -> Iterator[str]:
        """Yields content deltas."""
        with httpx.Client(timeout=120) as client:
            with client.stream(
                "POST",
                BASE_URL,
                headers=self.headers,
                json={
                    "model": model or self.cfg.answer_model,
                    "messages": messages,
                    "temperature": self.cfg.temperature,
                    "max_tokens": self.cfg.max_tokens,
                    "stream": True,
                },
            ) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line.startswith("data: "):
                        continue
                    payload = line[6:]
                    if payload.strip() == "[DONE]":
                        break
                    try:
                        delta = json.loads(payload)["choices"][0]["delta"]
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue
                    content = delta.get("content")
                    if content:
                        yield content
