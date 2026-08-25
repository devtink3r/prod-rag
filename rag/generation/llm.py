"""OpenRouter chat client (OpenAI-compatible) with SSE streaming."""

import json
from collections.abc import Iterator

import httpx

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
        with httpx.Client(timeout=120) as client:
            resp = client.post(
                BASE_URL,
                headers=self.headers,
                json={
                    "model": model or self.cfg.answer_model,
                    "messages": messages,
                    "temperature": self.cfg.temperature,
                    "max_tokens": max_tokens or self.cfg.max_tokens,
                },
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]

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
