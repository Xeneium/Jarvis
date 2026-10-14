"""API-key (Authorization header) support in the OpenAI-compat engine base."""

from __future__ import annotations

import json

import httpx
import pytest
import respx

from openjarvis.core.types import Message, Role
from openjarvis.engine.openai_compat_engines import (
    LMStudioEngine,
    OpenAICompatEngine,
    VLLMEngine,
    normalize_openai_base_url,
)

_CHAT_RESPONSE = {
    "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    "model": "m",
}


class TestAuthorizationHeader:
    def test_bearer_header_sent_when_api_key_set(self) -> None:
        engine = OpenAICompatEngine(host="http://testhost:9000", api_key="sk-test")
        with respx.mock:
            route = respx.post("http://testhost:9000/v1/chat/completions").mock(
                return_value=httpx.Response(200, json=_CHAT_RESPONSE)
            )
            engine.generate([Message(role=Role.USER, content="hi")], model="m")
        assert route.calls.last.request.headers["Authorization"] == "Bearer sk-test"

    def test_no_authorization_header_without_api_key(self) -> None:
        engine = OpenAICompatEngine(host="http://testhost:9000")
        with respx.mock:
            route = respx.post("http://testhost:9000/v1/chat/completions").mock(
                return_value=httpx.Response(200, json=_CHAT_RESPONSE)
            )
            engine.generate([Message(role=Role.USER, content="hi")], model="m")
        assert "authorization" not in route.calls.last.request.headers

    def test_health_check_sends_bearer_header(self) -> None:
        engine = OpenAICompatEngine(host="http://testhost:9000", api_key="sk-test")
        with respx.mock:
            route = respx.get("http://testhost:9000/v1/models").mock(
                return_value=httpx.Response(200, json={"data": []})
            )
            assert engine.health() is True
        assert route.calls.last.request.headers["Authorization"] == "Bearer sk-test"

    def test_env_var_fallback_sanitizes_hyphen(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # engine_id "openai-compat" must map to OPENAI_COMPAT_API_KEY —
        # shells cannot set hyphenated env-var names.
        monkeypatch.setenv("OPENAI_COMPAT_API_KEY", "sk-env")
        engine = OpenAICompatEngine(host="http://testhost:9000")
        with respx.mock:
            route = respx.get("http://testhost:9000/v1/models").mock(
                return_value=httpx.Response(200, json={"data": []})
            )
            engine.health()
        assert route.calls.last.request.headers["Authorization"] == "Bearer sk-env"

    def test_vllm_env_var_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("VLLM_API_KEY", "sk-vllm")
        engine = VLLMEngine(host="http://testhost:8000")
        with respx.mock:
            route = respx.get("http://testhost:8000/v1/models").mock(
                return_value=httpx.Response(200, json={"data": []})
            )
            engine.health()
        assert route.calls.last.request.headers["Authorization"] == "Bearer sk-vllm"

    def test_explicit_api_key_beats_env_var(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENAI_COMPAT_API_KEY", "sk-env")
        engine = OpenAICompatEngine(host="http://testhost:9000", api_key="sk-explicit")
        assert engine._api_key == "sk-explicit"

    def test_colibri_compatible_lmstudio_engine_uses_model_and_bearer_key(self) -> None:
        model_id = "colibri/qwen3-8b"
        engine = LMStudioEngine(host="http://127.0.0.1:8001", api_key="coli-test")
        with respx.mock:
            route = respx.post("http://127.0.0.1:8001/v1/chat/completions").mock(
                return_value=httpx.Response(200, json=_CHAT_RESPONSE)
            )
            engine.generate([Message(role=Role.USER, content="hi")], model=model_id)
        request = route.calls.last.request
        assert request.headers["Authorization"] == "Bearer coli-test"
        assert request.url.path == "/v1/chat/completions"
        assert json.loads(request.content)["model"] == model_id

    def test_lmstudio_model_list_uses_single_v1_prefix_and_optional_auth(self) -> None:
        host = normalize_openai_base_url("http://127.0.0.1:8001/v1")
        engine = LMStudioEngine(host=host, api_key="coli-test")
        with respx.mock:
            route = respx.get("http://127.0.0.1:8001/v1/models").mock(
                return_value=httpx.Response(
                    200, json={"data": [{"id": "colibri/qwen3-8b"}]}
                )
            )
            assert engine.list_models() == ["colibri/qwen3-8b"]
        request = route.calls.last.request
        assert request.url.path == "/v1/models"
        assert request.headers["Authorization"] == "Bearer coli-test"


class TestNormalizeOpenAIBaseUrl:
    @pytest.mark.parametrize(
        ("url", "expected"),
        [
            ("http://h:8000", "http://h:8000"),
            ("http://h:8000/", "http://h:8000"),
            ("http://h:8000/v1", "http://h:8000"),
            ("http://h:8000/v1/", "http://h:8000"),
            ("http://h:8000/gateway/v1", "http://h:8000/gateway"),
            # Only a literal trailing "/v1" is stripped — never other paths.
            ("http://h:8000/v1x", "http://h:8000/v1x"),
            ("http://h:8000/v2", "http://h:8000/v2"),
        ],
    )
    def test_normalization(self, url: str, expected: str) -> None:
        assert normalize_openai_base_url(url) == expected

    def test_engine_requests_have_single_v1_prefix(self) -> None:
        """End to end: a user-supplied .../v1 URL must not produce /v1/v1."""
        host = normalize_openai_base_url("http://testhost:9000/v1")
        engine = OpenAICompatEngine(host=host)
        with respx.mock:
            route = respx.get("http://testhost:9000/v1/models").mock(
                return_value=httpx.Response(200, json={"data": []})
            )
            assert engine.health() is True
        assert route.calls.last.request.url.path == "/v1/models"
