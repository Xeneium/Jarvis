"""Tests for Groq inference engine and online-fallback logic."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

import openjarvis.engine
from main import _load_env_file, _probe_groq_online
from openjarvis.cli._model_switch import resolve_chat_cli_model
from openjarvis.core.config import JarvisConfig
from openjarvis.core.registry import EngineRegistry
from openjarvis.engine.groq import GroqEngine


def test_groq_engine_registration():
    """Verify GroqEngine is registered in EngineRegistry under 'groq'."""
    EngineRegistry.register("groq")(GroqEngine)
    assert EngineRegistry.contains("groq")
    cls = EngineRegistry.get("groq")
    assert cls is GroqEngine
    assert cls.is_cloud is True


def test_groq_engine_init_and_defaults(monkeypatch):
    """Verify GroqEngine initializes with standard OpenAI-compatible endpoints."""
    monkeypatch.setenv("GROQ_API_KEY", "test-dummy-groq-key")
    engine = GroqEngine()
    assert engine._host == "https://api.groq.com/openai"
    assert engine._api_prefix == "/v1"
    assert engine._api_key == "test-dummy-groq-key"


def test_groq_engine_health_no_key(monkeypatch):
    """Verify health() returns False immediately if GROQ_API_KEY is missing."""
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    engine = GroqEngine()
    assert engine.health() is False


def test_groq_engine_health_with_mocked_success(monkeypatch):
    """Verify health() returns True when Groq responds 200."""
    monkeypatch.setenv("GROQ_API_KEY", "test-dummy-groq-key")
    engine = GroqEngine()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    with patch.object(engine._client, "get", return_value=mock_resp):
        assert engine.health() is True


def test_groq_engine_health_with_mocked_error(monkeypatch):
    """Verify health() returns False on non-200 or connection failure."""
    monkeypatch.setenv("GROQ_API_KEY", "test-dummy-groq-key")
    engine = GroqEngine()
    mock_resp = MagicMock()
    mock_resp.status_code = 401
    with patch.object(engine._client, "get", return_value=mock_resp):
        assert engine.health() is False

    with patch.object(engine._client, "get", side_effect=Exception("Timeout")):
        assert engine.health() is False


def test_groq_engine_list_models_filtering(monkeypatch):
    """Verify list_models() filters out non-chat models and prioritizes chat models."""
    monkeypatch.setenv("GROQ_API_KEY", "test-dummy-groq-key")
    engine = GroqEngine()
    mock_data = {
        "data": [
            {"id": "whisper-large-v3"},
            {"id": "openai/gpt-oss-120b"},
            {"id": "meta-llama/llama-prompt-guard-2-86m"},
            {"id": "qwen/qwen3.8-27b"},
            {"id": "canopylabs/orpheus-v1-english"},
        ]
    }
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = mock_data
    with patch.object(engine._client, "get", return_value=mock_resp):
        models = engine.list_models()
        assert "whisper-large-v3" not in models
        assert "meta-llama/llama-prompt-guard-2-86m" not in models
        assert "openai/gpt-oss-120b" in models
        assert "qwen/qwen3.8-27b" in models
        assert models[0] == "openai/gpt-oss-120b"


def test_groq_engine_can_serve(monkeypatch):
    """Verify can_serve() handles Groq models vs Ollama tags."""
    monkeypatch.setenv("GROQ_API_KEY", "test-dummy-groq-key")
    engine = GroqEngine()
    assert engine.can_serve("openai/gpt-oss-120b") is True
    assert engine.can_serve("qwen/qwen3.8-27b") is True
    assert engine.can_serve("llama-3.3-70b-versatile") is True
    assert engine.can_serve("gemma4:latest") is False
    assert engine.can_serve("qwen3-vl:4b") is False
    assert engine.can_serve("bge-m3:latest") is False
    assert engine.can_serve("any-model:tag") is False


def test_groq_engine_resolve_model_id_safety(monkeypatch):
    """Verify _resolve_model_id safely maps invalid or local Ollama tags to Groq model."""
    monkeypatch.setenv("GROQ_API_KEY", "test-dummy-groq-key")
    monkeypatch.setenv("GROQ_MODEL", "openai/gpt-oss-120b")
    engine = GroqEngine()

    assert engine._resolve_model_id("openai/gpt-oss-120b") == "openai/gpt-oss-120b"
    # Local Ollama tag must never be dispatched to Groq
    assert engine._resolve_model_id("qwen3-vl:4b") == "openai/gpt-oss-120b"
    assert engine._resolve_model_id("gemma4:latest") == "openai/gpt-oss-120b"


def test_probe_groq_online_success():
    """Verify _probe_groq_online detects successful connection."""
    mock_response = MagicMock()
    mock_response.status = 200
    mock_response.read.return_value = b'{"data": [{"id": "openai/gpt-oss-120b"}, {"id": "qwen/qwen3.8-27b"}]}'

    with patch("urllib.request.urlopen") as mock_urlopen:
        mock_urlopen.return_value.__enter__.return_value = mock_response
        online, model = _probe_groq_online("gsk_valid_key")
        assert online is True
        assert model == "openai/gpt-oss-120b"


def test_probe_groq_online_custom_model(monkeypatch):
    """Verify _probe_groq_online respects GROQ_MODEL in .env if available."""
    monkeypatch.setenv("GROQ_MODEL", "qwen/qwen3.8-27b")
    mock_response = MagicMock()
    mock_response.status = 200
    mock_response.read.return_value = b'{"data": [{"id": "openai/gpt-oss-120b"}, {"id": "qwen/qwen3.8-27b"}]}'

    with patch("urllib.request.urlopen") as mock_urlopen:
        mock_urlopen.return_value.__enter__.return_value = mock_response
        online, model = _probe_groq_online("gsk_valid_key")
        assert online is True
        assert model == "qwen/qwen3.8-27b"


def test_probe_groq_online_failure():
    """Verify _probe_groq_online gracefully catches timeouts and errors."""
    with patch("urllib.request.urlopen", side_effect=Exception("Network down")):
        online, model = _probe_groq_online("gsk_valid_key")
        assert online is False
        assert model is None

    online, model = _probe_groq_online("")
    assert online is False
    assert model is None


def test_resolve_chat_cli_model_groq(monkeypatch):
    """Verify resolve_chat_cli_model selects Groq model when engine is groq."""
    config = JarvisConfig()
    engine = GroqEngine(api_key="gsk_test123")
    with patch.object(engine, "list_models", return_value=["openai/gpt-oss-120b", "qwen/qwen3.8-27b"]):
        resolved = resolve_chat_cli_model(
            console=MagicMock(),
            config=config,
            engine=engine,
            engine_name="groq",
            cli_model=None,
            chat_variant="chat",
        )
        assert resolved in ("openai/gpt-oss-120b", "qwen/qwen3.8-27b")


def test_resolve_chat_cli_model_groq_rejects_stale_qwen3_vl(monkeypatch):
    """Verify resolve_chat_cli_model will never return qwen3-vl:4b for groq even if set in config."""
    config = JarvisConfig()
    config.intelligence.default_model = "qwen3-vl:4b"
    engine = GroqEngine(api_key="gsk_test123")
    monkeypatch.setenv("GROQ_MODEL", "openai/gpt-oss-120b")
    resolved = resolve_chat_cli_model(
        console=MagicMock(),
        config=config,
        engine=engine,
        engine_name="groq",
        cli_model=None,
        chat_variant="chat",
    )
    assert resolved == "openai/gpt-oss-120b"


def test_resolve_chat_cli_model_ollama_fallback(monkeypatch):
    """Verify resolve_chat_cli_model preserves gemma4:latest when engine is ollama."""
    config = JarvisConfig()
    engine = MagicMock()
    engine.list_models.return_value = ["gemma4:latest"]
    resolved = resolve_chat_cli_model(
        console=MagicMock(),
        config=config,
        engine=engine,
        engine_name="ollama",
        cli_model=None,
        chat_variant="chat",
    )
    assert resolved == "gemma4:latest"


def test_groq_engine_recovers_failed_generation(monkeypatch):
    """Verify GroqEngine recovers from HTTP 400 tool_use_failed with failed_generation."""
    import json
    import httpx
    from openjarvis.core.types import Message, Role

    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    engine = GroqEngine()

    error_body = {
        "error": {
            "message": "Tool choice is none, but model called a tool",
            "type": "invalid_request_error",
            "code": "tool_use_failed",
            "failed_generation": json.dumps({
                "name": "deep_view_search",
                "arguments": {"action": "search", "query": "AI", "limit": 5},
            }),
        }
    }
    mock_resp = httpx.Response(
        status_code=400,
        json=error_body,
        request=httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions"),
    )

    with patch.object(engine._client, "post", return_value=mock_resp):
        res = engine.generate(
            [Message(role=Role.USER, content="What is the latest news in AI?")],
            model="openai/gpt-oss-120b",
        )
        assert res["finish_reason"] == "tool_calls"
        assert len(res["tool_calls"]) == 1
        assert res["tool_calls"][0]["name"] == "deep_view_search"
        assert "Action: deep_view_search" in res["content"]
        assert "Action Input:" in res["content"]


def test_native_react_agent_handles_recovered_tool_call():
    """Verify NativeReActAgent uses recovered tool_calls when Action missing from content."""
    from openjarvis.agents.native_react import NativeReActAgent
    from openjarvis.tools.deep_view import TheDeepViewTool

    mock_engine = MagicMock()
    # Turn 1: returns recovered tool call
    mock_engine.generate.side_effect = [
        {
            "content": "",
            "tool_calls": [
                {
                    "id": "groq_call_1",
                    "name": "deep_view_search",
                    "arguments": '{"action": "search", "query": "AI", "limit": 1}',
                }
            ],
            "finish_reason": "tool_calls",
            "usage": {},
        },
        # Turn 2: returns final answer
        {
            "content": "Final Answer: The latest news from The Deep View is GPT-6 launch.",
            "tool_calls": [],
            "finish_reason": "stop",
            "usage": {},
        },
    ]

    tool = TheDeepViewTool()
    with patch.object(tool, "execute") as mock_exec:
        from openjarvis.core.types import ToolResult
        mock_exec.return_value = ToolResult(
            tool_name="deep_view_search",
            content="[Sample Story](https://www.thedeepview.com/article/1)",
            success=True,
        )
        agent = NativeReActAgent(mock_engine, "openai/gpt-oss-120b", tools=[tool], max_turns=3)
        res = agent.run("What is the latest news in AI?")
        assert "The latest news from The Deep View is GPT-6 launch." in res.content
        assert res.turns == 2
        mock_exec.assert_called_once()

