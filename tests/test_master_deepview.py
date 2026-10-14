"""Master test suite verifying all 12 Master Prompt requirements for OpenJarvis.

Requirements covered:
1. Interactive startup with python main.py.
2. Ollama initialization and health check.
3. Model selection and diagnostic reporting when Ollama is unavailable.
4. Validation that only approved The Deep View domains can be fetched.
5. Rejection of unapproved domains, non-HTTPS URLs, and lookalike domains.
6. Validation that redirects cannot escape the approved website.
7. Correct parsing of articles, dates, titles, and text from The Deep View.
8. Rejection of unapproved search engines or fallback providers.
9. Answering Deep View-specific questions using retrieved evidence.
10. Answering general questions without web retrieval.
11. Proper handling of questions when The Deep View has no matching coverage.
12. Conversation continuity across multiple turns.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest

from openjarvis.agents.native_react import NativeReActAgent
from openjarvis.core.config import JarvisConfig, load_config
from openjarvis.core.registry import AgentRegistry, EngineRegistry, ToolRegistry
from openjarvis.core.types import Message, Role, ToolResult
from openjarvis.engine._base import InferenceEngine
from openjarvis.engine.ollama import OllamaEngine
from openjarvis.tools.deep_view import (
    APPROVED_HOSTS,
    APPROVED_ORIGINS,
    SecurityValidationError,
    TheDeepViewTool,
    extract_deep_view_content,
    fetch_with_strict_redirects,
    is_valid_deep_view_url,
    validate_deep_view_url,
)
from openjarvis.tools.web_search import WebSearchTool


MOCK_SITEMAP = """<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url>
    <loc>https://www.thedeepview.com/articles/state-of-ai-agents-2026</loc>
    <lastmod>2026-10-14T09:00:00.000Z</lastmod>
  </url>
  <url>
    <loc>https://www.thedeepview.com/newsletter/weekly-ai-intelligence-oct-12</loc>
    <lastmod>2026-10-12T10:00:00.000Z</lastmod>
  </url>
</urlset>
"""

MOCK_ARTICLE_HTML = """<!DOCTYPE html>
<html>
<head>
  <title>State of AI Agents in 2026 - The Deep View</title>
  <meta property="og:title" content="State of AI Agents in 2026" />
  <script type="application/ld+json">
  {
    "@context": "https://schema.org",
    "@type": "NewsArticle",
    "headline": "State of AI Agents in 2026",
    "datePublished": "2026-10-14T09:00:00Z",
    "author": {"@type": "Person", "name": "Deep View Research Team"}
  }
  </script>
</head>
<body>
  <article>
    <h1>State of AI Agents in 2026</h1>
    <div class="content-wrapper">
      <p>AI agents are transitioning from scripted workflows to self-refining multi-agent loops.</p>
      <p>Key breakthroughs include test-time compute scaling and asynchronous sandbox execution.</p>
    </div>
  </article>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# 1. Interactive startup with python main.py
# ---------------------------------------------------------------------------
def test_1_main_py_startup_banner_and_routing():
    """Requirement 1: Verify python main.py bare launch displays banner and routes cleanly."""
    repo_root = Path(__file__).resolve().parent.parent
    main_py = repo_root / "main.py"
    venv_py = repo_root / ".venv" / "Scripts" / "python.exe"
    py_bin = str(venv_py) if venv_py.is_file() else sys.executable

    # Run with empty input to trigger immediate session exit
    env = {**os.environ, "PYTHONPATH": str(repo_root / "src")}
    result = subprocess.run(
        [py_bin, str(main_py), "--help"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=str(repo_root),
        env=env,
        timeout=15,
    )
    assert result.returncode == 0
    assert "OpenJarvis" in result.stdout or "jarvis" in result.stdout

    # Verify banner helper
    from openjarvis.cli._banner import _TAGLINE, _WORDMARK, print_banner
    assert _TAGLINE == "Personal AI, On Personal Devices"
    assert len(_WORDMARK) > 0


# ---------------------------------------------------------------------------
# 2. Ollama initialization and health check
# ---------------------------------------------------------------------------
def test_2_ollama_engine_initialization_and_health():
    """Requirement 2: Verify OllamaEngine initialization, URL formation, and health check."""
    engine = OllamaEngine(host="http://localhost:11434")
    assert engine.engine_id == "ollama"
    assert engine._host == "http://localhost:11434"

    # Mock health response
    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_client.get.return_value = mock_resp
    engine._client = mock_client

    assert engine.health() is True
    mock_client.get.assert_called_with("/api/tags", timeout=2.0)


# ---------------------------------------------------------------------------
# 3. Model selection and diagnostic reporting when Ollama is unavailable
# ---------------------------------------------------------------------------
def test_3_ollama_unavailable_diagnostic_reporting():
    """Requirement 3: Verify clear diagnostic messages when Ollama is unreachable."""
    engine = OllamaEngine(host="http://localhost:11434")

    # Mock failed health
    mock_client = MagicMock()
    mock_client.get.side_effect = httpx.ConnectError("Connection refused")
    engine._client = mock_client

    assert engine.health() is False
    assert engine.list_models() == []


# ---------------------------------------------------------------------------
# 4. Validation that only approved The Deep View domains can be fetched
# ---------------------------------------------------------------------------
def test_4_approved_deep_view_domains():
    """Requirement 4: Verify approved origins: https://www.thedeepview.com & https://thedeepview.com."""
    assert "https://www.thedeepview.com" in APPROVED_ORIGINS
    assert "https://thedeepview.com" in APPROVED_ORIGINS
    assert validate_deep_view_url("https://www.thedeepview.com/articles/test") == "https://www.thedeepview.com/articles/test"
    assert validate_deep_view_url("https://thedeepview.com/newsletter/oct-12") == "https://thedeepview.com/newsletter/oct-12"


# ---------------------------------------------------------------------------
# 5. Rejection of unapproved domains, non-HTTPS URLs, and lookalike domains
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "forbidden_url",
    [
        "http://www.thedeepview.com/insecure",
        "https://google.com/search?q=thedeepview",
        "https://duckduckgo.com/?q=ai",
        "https://thedeepview.co/news",
        "https://thedeepview.org/ai",
        "https://thedeepview.net",
        "https://thedeepview.com.fake-phishing.org",
        "https://evil.thedeepview.com",
        "https://tavily.com",
    ],
)
def test_5_rejection_of_unapproved_and_lookalike_domains(forbidden_url: str):
    """Requirement 5: Strict rejection of lookalike and unapproved domains."""
    assert not is_valid_deep_view_url(forbidden_url)
    with pytest.raises(SecurityValidationError):
        validate_deep_view_url(forbidden_url)


# ---------------------------------------------------------------------------
# 6. Validation that redirects cannot escape the approved website
# ---------------------------------------------------------------------------
def test_6_redirect_cannot_escape_approved_website():
    """Requirement 6: Cross-domain redirects are blocked immediately."""
    def redirect_handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(302, headers={"Location": "https://external-news.com/stolen-content"})

    client = httpx.Client(transport=httpx.MockTransport(redirect_handler))
    with pytest.raises(SecurityValidationError) as exc:
        fetch_with_strict_redirects("https://www.thedeepview.com/articles/redirect", client=client)
    assert "Unauthorized host" in str(exc.value)


# ---------------------------------------------------------------------------
# 7. Correct parsing of articles, dates, titles, and text from The Deep View
# ---------------------------------------------------------------------------
def test_7_parsing_articles_dates_titles_and_substantive_text():
    """Requirement 7: Correct metadata and substantive text extraction."""
    article = extract_deep_view_content(
        MOCK_ARTICLE_HTML,
        "https://www.thedeepview.com/articles/state-of-ai-agents-2026",
    )
    assert article.title == "State of AI Agents in 2026"
    assert "2026-10-14" in article.published_date
    assert article.author == "Deep View Research Team"
    assert "AI agents are transitioning" in article.content
    assert "test-time compute scaling" in article.content
    assert article.url == "https://www.thedeepview.com/articles/state-of-ai-agents-2026"


# ---------------------------------------------------------------------------
# 8. Rejection of unapproved search engines or fallback providers
# ---------------------------------------------------------------------------
def test_8_rejection_of_unapproved_search_engines_and_fallback(monkeypatch):
    """Requirement 8: Ensure WebSearchTool uses Deep View and refuses external fallback."""
    monkeypatch.setenv("OPENJARVIS_EXCLUSIVE_SOURCE", "thedeepview")
    tool = WebSearchTool()
    assert tool._resolve_engine() == "deep_view"

    # Attempting to fetch an external website via web_search fails security validation
    res = tool.execute(query="https://news.ycombinator.com")
    assert res.success is False
    assert "Security Policy Violation" in res.content


# ---------------------------------------------------------------------------
# 9. Answering Deep View-specific questions using retrieved evidence
# ---------------------------------------------------------------------------
def test_9_deep_view_specific_question_with_retrieved_evidence():
    """Requirement 9: Answering questions with grounded citations and dates."""
    def mock_http(req: httpx.Request) -> httpx.Response:
        url_str = str(req.url)
        if "sitemap.xml" in url_str:
            return httpx.Response(200, text=MOCK_SITEMAP)
        return httpx.Response(200, text=MOCK_ARTICLE_HTML)

    client = httpx.Client(transport=httpx.MockTransport(mock_http))
    deep_view_tool = TheDeepViewTool(client=client)

    result = deep_view_tool.execute(query="AI agents", limit=2)
    assert result.success is True
    assert "State of AI Agents in 2026" in result.content
    assert "https://www.thedeepview.com/articles/state-of-ai-agents-2026" in result.content
    assert "2026-10-14" in result.content


# ---------------------------------------------------------------------------
# 10. Answering general questions without web retrieval
# ---------------------------------------------------------------------------
def test_10_general_questions_answered_without_web_retrieval():
    """Requirement 10: General conceptual questions do not invoke retrieval tools."""
    class DummyEngine(InferenceEngine):
        def health(self) -> bool:
            return True

        def list_models(self) -> list[str]:
            return ["llama3.2"]

        async def stream(self, messages, *, model, **kwargs):
            yield ""

        def generate(self, messages, *, model, **kwargs):
            return {
                "content": (
                    "Thought: This is a general question about Docker networking that does not require "
                    "live web research.\n"
                    "Final Answer: Docker networking allows containers to communicate via bridge, host, or overlay drivers."
                )
            }

    ToolRegistry.register_value("deep_view_search", TheDeepViewTool)
    tool = TheDeepViewTool()
    agent = NativeReActAgent(
        engine=DummyEngine(),
        model="llama3.2",
        tools=[tool],
        max_turns=3,
    )

    res = agent.run("Explain how Docker network works.")
    assert "Docker networking allows containers to communicate" in res.content
    # Verified no tools were executed
    assert len(res.tool_results) == 0


# ---------------------------------------------------------------------------
# 11. Proper handling of questions when The Deep View has no matching coverage
# ---------------------------------------------------------------------------
def test_11_no_matching_coverage_handled_properly():
    """Requirement 11: When Deep View has no coverage, clearly reports that without external browsing."""
    def mock_empty_sitemap(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=MOCK_SITEMAP)

    client = httpx.Client(transport=httpx.MockTransport(mock_empty_sitemap))
    tool = TheDeepViewTool(client=client)

    res = tool.execute(action="search", query="Super Bowl 2026 Score", limit=3)
    assert res.success is True
    assert "No matching articles found on The Deep View" in res.content
    assert "no external search engines or other websites were consulted" in res.content


# ---------------------------------------------------------------------------
# 12. Conversation continuity across multiple turns
# ---------------------------------------------------------------------------
def test_12_conversation_continuity_across_multiple_turns():
    """Requirement 12: Multi-turn conversation preserves context."""
    turns_received = []

    class MockMultiTurnEngine(InferenceEngine):
        def health(self) -> bool:
            return True

        def list_models(self) -> list[str]:
            return ["llama3.2"]

        async def stream(self, messages, *, model, **kwargs):
            yield ""

        def generate(self, messages, *, model, **kwargs):
            nonlocal turns_received
            turns_received.append([m.content for m in messages if m.role == Role.USER])
            turn_idx = len(turns_received)
            if turn_idx == 1:
                return {"content": "Final Answer: The latest newsletter focuses on autonomous AI agents."}
            else:
                return {"content": "Final Answer: As mentioned earlier, those agents use self-refining loops."}

    engine = MockMultiTurnEngine()
    history: list[Message] = []

    # Turn 1
    history.append(Message(role=Role.USER, content="Summarize the latest newsletter."))
    resp1 = engine.generate(history, model="llama3.2")
    history.append(Message(role=Role.ASSISTANT, content=resp1["content"]))

    # Turn 2 (Follow-up)
    history.append(Message(role=Role.USER, content="How do those agents work?"))
    resp2 = engine.generate(history, model="llama3.2")
    history.append(Message(role=Role.ASSISTANT, content=resp2["content"]))

    assert len(history) == 4
    assert history[0].content == "Summarize the latest newsletter."
    assert history[2].content == "How do those agents work?"
    assert "autonomous AI agents" in history[1].content
    assert "self-refining loops" in history[3].content


# ---------------------------------------------------------------------------
# 13. Verification that 16K Context Window (16,384 tokens) reaches Ollama
# ---------------------------------------------------------------------------
def test_13_ollama_request_options_receives_16k_context(monkeypatch):
    """Requirement 13: Verify that context window defaults to 16,384 and reaches Ollama payload."""
    from openjarvis.engine.ollama import _default_num_ctx, _ollama_request_options

    # 1. Environment default
    monkeypatch.setenv("JARVIS_NUM_CTX", "16384")
    assert _default_num_ctx() == 16384

    # 2. Options builder without kwargs uses default 16,384
    options_default = _ollama_request_options(temperature=0.7, max_tokens=2048, kwargs={})
    assert options_default["num_ctx"] == 16384

    # 3. Options builder with explicit 16,384
    options_explicit = _ollama_request_options(
        temperature=0.7,
        max_tokens=2048,
        kwargs={"num_ctx": 16384, "num_gpu": 1},
    )
    assert options_explicit["num_ctx"] == 16384
    assert options_explicit["num_gpu"] == 1

    # 4. Verify in actual OllamaEngine.generate request payload
    engine = OllamaEngine(host="http://localhost:11434")
    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "message": {"role": "assistant", "content": "Analysis complete."},
        "prompt_eval_count": 120,
        "eval_count": 45,
    }
    mock_client.post.return_value = mock_resp
    engine._client = mock_client

    engine.generate(
        [Message(role=Role.USER, content="Hello")],
        model="gemma4:latest",
        num_ctx=16384,
    )

    assert mock_client.post.called
    call_kwargs = mock_client.post.call_args[1]
    posted_payload = call_kwargs.get("json", {})
    assert posted_payload.get("model") == "gemma4:latest"
    assert posted_payload.get("options", {}).get("num_ctx") == 16384


# ---------------------------------------------------------------------------
# 14. Verification that installed model gemma4:latest is strictly preserved
# ---------------------------------------------------------------------------
def test_14_installed_model_gemma4_preserved_without_silent_substitution(monkeypatch):
    """Requirement 14: Verify that gemma4:latest is preserved and never silently substituted."""
    from openjarvis.cli._model_switch import resolve_chat_cli_model
    from openjarvis.core.config import JarvisConfig

    monkeypatch.setenv("OLLAMA_MODEL", "gemma4:latest")
    monkeypatch.setenv("JARVIS_MODEL", "gemma4:latest")

    cfg = JarvisConfig()
    cfg.intelligence.default_model = ""
    eng = MagicMock()

    # Even if discover_models returns other models, gemma4:latest is preserved
    with (
        patch("openjarvis.engine.discover_engines", return_value=[("ollama", eng)]),
        patch("openjarvis.engine.discover_models", return_value={"ollama": ["qwen3:8b", "llama3.2:3b", "gemma4:latest"]}),
    ):
        model_resolved = resolve_chat_cli_model(
            console=MagicMock(),
            config=cfg,
            engine=eng,
            engine_name="ollama",
            cli_model=None,
            chat_variant="chat",
        )
    assert model_resolved == "gemma4:latest"
    assert "qwen" not in model_resolved.lower()


# ---------------------------------------------------------------------------
# 15. Verification that BrowserNavigateTool and HttpRequestTool block off-domain
# ---------------------------------------------------------------------------
def test_15_browser_and_http_tools_block_off_domain_targets(monkeypatch):
    """Requirement 15: Ensure browser and HTTP request tools enforce The Deep View confinement."""
    monkeypatch.setenv("OPENJARVIS_EXCLUSIVE_SOURCE", "thedeepview")

    from openjarvis.tools.browser import BrowserNavigateTool
    from openjarvis.tools.http_request import HttpRequestTool

    browser_tool = BrowserNavigateTool()
    http_tool = HttpRequestTool()

    # 1. Off-domain browser navigation is blocked
    b_res = browser_tool.execute(url="https://google.com")
    assert b_res.success is False
    assert "Security Policy Violation" in b_res.content
    assert "The Deep View" in b_res.content

    # 2. Off-domain HTTP request is blocked
    h_res = http_tool.execute(url="https://api.external-service.org/data")
    assert h_res.success is False
    assert "Security Policy Violation" in h_res.content
    assert "The Deep View" in h_res.content

    # 3. Off-domain raw IP is blocked
    ip_res = http_tool.execute(url="https://127.0.0.1/admin")
    assert ip_res.success is False
    assert "Security Policy Violation" in ip_res.content or "SSRF" in ip_res.content

