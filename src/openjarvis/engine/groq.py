"""Groq inference engine — ultra-fast LPU inference backend."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Sequence

import httpx

from openjarvis.core.registry import EngineRegistry
from openjarvis.engine._base import EngineConnectionError
from openjarvis.engine._openai_compat import _OpenAICompatibleEngine

logger = logging.getLogger(__name__)

# Preferred models in priority order
DEFAULT_GROQ_MODELS = [
    "openai/gpt-oss-120b",
    "qwen/qwen3.8-27b",
    "openai/gpt-oss-20b",
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
]


@EngineRegistry.register("groq")
class GroqEngine(_OpenAICompatibleEngine):
    """Groq LPU inference engine with OpenAI-compatible API.

    Requires ``GROQ_API_KEY`` from https://console.groq.com.
    Endpoint defaults to https://api.groq.com/openai/v1.
    """

    engine_id: str = "groq"
    _default_host: str = "https://api.groq.com/openai"
    _api_prefix: str = "/v1"
    is_cloud: bool = True

    def __init__(
        self,
        host: str | None = None,
        *,
        api_key: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        key = api_key or os.environ.get("GROQ_API_KEY") or None
        super().__init__(host=host, api_key=key, timeout=timeout)

    def health(self) -> bool:
        """Check if Groq API is reachable and API key is valid."""
        if not self._api_key:
            return False
        try:
            resp = self._client.get(f"{self._api_prefix}/models", timeout=3.0)
            return resp.status_code == 200
        except Exception as exc:
            logger.debug("Groq health check failed at %s: %s", self._host, exc)
            return False

    def list_models(self) -> List[str]:
        """Return available chat models from Groq."""
        if not self._api_key:
            return []
        try:
            resp = self._client.get(f"{self._api_prefix}/models", timeout=5.0)
            if resp.status_code != 200:
                return []
            data = resp.json().get("data", [])
            raw_ids = [m["id"] for m in data if isinstance(m, dict) and "id" in m]

            # Filter out non-chat models (e.g. whisper, prompt guard)
            chat_models = [
                m
                for m in raw_ids
                if not m.startswith("whisper") and "guard" not in m.lower()
            ]

            # Sort so preferred models appear first
            ordered: List[str] = []
            for pref in DEFAULT_GROQ_MODELS:
                if pref in chat_models and pref not in ordered:
                    ordered.append(pref)
            for m in chat_models:
                if m not in ordered:
                    ordered.append(m)

            return ordered if ordered else raw_ids
        except Exception as exc:
            logger.warning("Failed to list models from Groq: %s", exc)
            return []

    def can_serve(self, model: str) -> bool:
        """Check if this engine can serve the given model."""
        if not self._api_key:
            return False
        # If model is an Ollama tag, Groq cannot serve it
        if "gemma4" in model.lower():
            return False
        return True


__all__ = ["GroqEngine", "DEFAULT_GROQ_MODELS"]
