"""Unit and integration tests for The Deep View exclusive research tool."""

from __future__ import annotations

import httpx
import pytest

from openjarvis.core.registry import ToolRegistry
from openjarvis.core.types import ToolCall
from openjarvis.tools.deep_view import (
    ALLOWED_CONTENT_TYPES,
    APPROVED_HOSTS,
    APPROVED_ORIGINS,
    MAX_PAGE_BYTES,
    DeepViewArticle,
    SecurityValidationError,
    TheDeepViewRetriever,
    TheDeepViewTool,
    extract_deep_view_content,
    fetch_with_strict_redirects,
    is_deep_view_origin,
    is_valid_deep_view_url,
    validate_deep_view_url,
    verify_deep_view_network_safety,
)
from openjarvis.tools.web_search import WebSearchTool


SAMPLE_SITEMAP_XML = """<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url>
    <loc>https://www.thedeepview.com/</loc>
    <lastmod>2026-10-14T08:00:00.000Z</lastmod>
  </url>
  <url>
    <loc>https://www.thedeepview.com/articles/openai-new-reasoning-model</loc>
    <lastmod>2026-10-13T12:00:00.000Z</lastmod>
  </url>
  <url>
    <loc>https://www.thedeepview.com/newsletter/autonomous-agents-breakthrough</loc>
    <lastmod>2026-10-12T15:30:00.000Z</lastmod>
  </url>
  <url>
    <loc>https://thedeepview.com/articles/nvidia-next-gen-ai-chips</loc>
    <lastmod>2026-10-10T09:00:00.000Z</lastmod>
  </url>
</urlset>
"""

SAMPLE_ARTICLE_HTML = """<!DOCTYPE html>
<html>
<head>
  <title>OpenAI Releases New Reasoning Model - The Deep View</title>
  <meta property="og:title" content="OpenAI Releases New Reasoning Model" />
  <meta property="og:url" content="https://www.thedeepview.com/articles/openai-new-reasoning-model" />
  <script type="application/ld+json">
  {
    "@context": "https://schema.org",
    "@type": "NewsArticle",
    "headline": "OpenAI Releases New Reasoning Model",
    "datePublished": "2026-10-13T12:00:00Z",
    "author": {
      "@type": "Person",
      "name": "Alex Chen"
    }
  }
  </script>
</head>
<body>
  <header><nav>Navigation links to ignore</nav></header>
  <article class="max-w-none">
    <h1>OpenAI Releases New Reasoning Model</h1>
    <div class="content-wrapper">
      <p>OpenAI has introduced a major update to its reasoning models designed for autonomous problem-solving.</p>
      <p>The new framework achieves state-of-the-art results on software engineering benchmarks and competitive coding evaluations.</p>
      <p>Researchers highlighted that the key breakthrough involves reinforcement learning over long-horizon traces.</p>
    </div>
  </article>
  <footer>Copyright 2026 The Deep View. All rights reserved.</footer>
</body>
</html>
"""


def test_approved_origins_and_hosts():
    assert "https://www.thedeepview.com" in APPROVED_ORIGINS
    assert "https://thedeepview.com" in APPROVED_ORIGINS
    assert "www.thedeepview.com" in APPROVED_HOSTS
    assert "thedeepview.com" in APPROVED_HOSTS


def test_valid_deep_view_urls():
    assert is_valid_deep_view_url("https://www.thedeepview.com")
    assert is_valid_deep_view_url("https://thedeepview.com/newsletter")
    assert is_valid_deep_view_url("https://www.thedeepview.com/articles/ai-agent-breakthrough?id=123")
    assert validate_deep_view_url("https://thedeepview.com/page").startswith("https://thedeepview.com")


@pytest.mark.parametrize(
    "invalid_url",
    [
        "http://www.thedeepview.com",             # Insecure HTTP
        "http://thedeepview.com",                 # Insecure HTTP
        "https://google.com",                     # External search engine
        "https://bing.com",                       # External search engine
        "https://duckduckgo.com",                 # External search engine
        "https://tavily.com",                     # External provider
        "https://you.com",                        # External provider
        "https://techcrunch.com",                 # External news
        "https://thedeepview.org",                # Lookalike TLD
        "https://thedeepview.co",                 # Lookalike TLD
        "https://thedeepview.net",                # Lookalike TLD
        "https://thedeepview.com.attacker.com",   # Subdomain hijacking lookalike
        "https://fake.thedeepview.com",           # Unapproved subdomain
        "https://user:pass@www.thedeepview.com",  # Embedded credentials
        "https://www.thedeepview.com:8080/test",  # Non-standard port
        "ftp://www.thedeepview.com",              # Disallowed scheme
        "javascript:alert(1)",                    # XSS scheme
        "",                                       # Empty string
    ],
)
def test_rejected_unapproved_urls(invalid_url: str):
    assert not is_valid_deep_view_url(invalid_url)
    with pytest.raises(SecurityValidationError):
        validate_deep_view_url(invalid_url)


def test_redirect_stays_within_deep_view():
    """Verify that redirects staying on thedeepview.com succeed."""
    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url) == "https://thedeepview.com/newsletter":
            return httpx.Response(301, headers={"Location": "https://www.thedeepview.com/newsletter"})
        return httpx.Response(200, text="Final Destination Content")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    content, final_url = fetch_with_strict_redirects("https://thedeepview.com/newsletter", client=client)
    assert content == "Final Destination Content"
    assert final_url == "https://www.thedeepview.com/newsletter"


def test_redirect_escaping_to_external_domain_blocked():
    """Verify that a redirect attempting to leave The Deep View is blocked immediately."""
    def malicious_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(302, headers={"Location": "https://attacker.com/leak"})

    client = httpx.Client(transport=httpx.MockTransport(malicious_handler))
    with pytest.raises(SecurityValidationError) as exc_info:
        fetch_with_strict_redirects("https://www.thedeepview.com/articles/test", client=client)
    assert "Unauthorized host 'attacker.com'" in str(exc_info.value)


def test_extract_deep_view_content():
    """Verify HTML extraction of metadata and clean article text."""
    article = extract_deep_view_content(
        SAMPLE_ARTICLE_HTML,
        "https://www.thedeepview.com/articles/openai-new-reasoning-model",
    )
    assert article.title == "OpenAI Releases New Reasoning Model"
    assert "2026-10-13" in article.published_date
    assert article.author == "Alex Chen"
    assert "OpenAI has introduced a major update" in article.content
    assert "reinforcement learning over long-horizon traces" in article.content
    assert "Navigation links to ignore" not in article.content
    assert "Copyright 2026" not in article.content
    assert len(article.summary) > 20


def test_deep_view_retriever_search_and_newsletters():
    """Verify sitemap search and newsletter retrieval with mocked transport."""
    def sitemap_and_page_handler(request: httpx.Request) -> httpx.Response:
        url_str = str(request.url)
        if url_str == "https://www.thedeepview.com/sitemap.xml":
            return httpx.Response(200, text=SAMPLE_SITEMAP_XML)
        elif "openai" in url_str:
            return httpx.Response(200, text=SAMPLE_ARTICLE_HTML)
        return httpx.Response(200, text="<html><body><article><p>Generic newsletter text.</p></article></body></html>")

    client = httpx.Client(transport=httpx.MockTransport(sitemap_and_page_handler))
    retriever = TheDeepViewRetriever(client=client)

    # Search for OpenAI
    results = retriever.search(query="OpenAI", limit=2)
    assert len(results) > 0
    top = results[0]
    assert "OpenAI" in top["title"]
    assert "https://www.thedeepview.com/articles/openai-new-reasoning-model" in top["url"]

    # Search for latest newsletters
    newsletters = retriever.get_latest_newsletters(limit=2)
    assert len(newsletters) > 0
    assert "autonomous-agents-breakthrough" in newsletters[0]["url"]


def test_deep_view_tool_registry():
    """Verify tool is properly registered in ToolRegistry."""
    ToolRegistry.register_value("deep_view_search", TheDeepViewTool)
    ToolRegistry.register_value("deep_view_research", TheDeepViewTool)
    assert ToolRegistry.contains("deep_view_search")
    assert ToolRegistry.contains("deep_view_research")
    tool = ToolRegistry.get("deep_view_search")()
    assert tool.tool_id == "deep_view_search"
    assert tool.spec.name == "deep_view_search"


def test_deep_view_tool_execution():
    """Verify tool execution for search, latest_newsletters, and read_article actions."""
    def mock_handler(request: httpx.Request) -> httpx.Response:
        url_str = str(request.url)
        if "sitemap.xml" in url_str:
            return httpx.Response(200, text=SAMPLE_SITEMAP_XML)
        return httpx.Response(200, text=SAMPLE_ARTICLE_HTML)

    client = httpx.Client(transport=httpx.MockTransport(mock_handler))
    tool = TheDeepViewTool(client=client)

    # Action 1: Search
    res_search = tool.execute(action="search", query="OpenAI", limit=3)
    assert res_search.success is True
    assert "OpenAI" in res_search.content
    assert "https://www.thedeepview.com" in res_search.content

    # Action 2: Latest newsletters
    res_news = tool.execute(action="latest_newsletters", limit=2)
    assert res_news.success is True
    assert "The Deep View - Latest Newsletters" in res_news.content

    # Action 3: Read article
    res_read = tool.execute(
        action="read_article",
        url="https://www.thedeepview.com/articles/openai-new-reasoning-model",
    )
    assert res_read.success is True
    assert "OpenAI Releases New Reasoning Model" in res_read.content
    assert "Alex Chen" in res_read.content
    assert "reinforcement learning" in res_read.content


def test_web_search_tool_delegates_to_deep_view(monkeypatch):
    """Verify WebSearchTool strictly uses The Deep View when configured in exclusive mode."""
    monkeypatch.setenv("OPENJARVIS_EXCLUSIVE_SOURCE", "thedeepview")

    def mock_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=SAMPLE_SITEMAP_XML)

    tool = WebSearchTool()
    assert tool._resolve_engine() == "deep_view"

    # Block external URL in web_search
    res_bad_url = tool.execute(query="https://unauthorized-domain.com/news")
    assert res_bad_url.success is False
    assert "Security Policy Violation" in res_bad_url.content


def test_is_deep_view_origin_alias():
    """Verify is_deep_view_origin acts as a canonical alias for domain validity."""
    assert is_deep_view_origin("https://www.thedeepview.com/articles/ai") is True
    assert is_deep_view_origin("https://thedeepview.com/newsletter/latest") is True
    assert is_deep_view_origin("https://evil.thedeepview.com/x") is False
    assert is_deep_view_origin("https://google.com") is False


@pytest.mark.parametrize(
    "prohibited_target",
    [
        "https://127.0.0.1/articles/ai",
        "https://10.0.0.1/admin",
        "https://192.168.1.50/login",
        "https://169.254.169.254/latest/meta-data/",
        "https://[::1]/internal",
    ],
)
def test_verify_deep_view_network_safety_rejects_ip_literals(prohibited_target: str):
    """Verify that verify_deep_view_network_safety unconditionally rejects raw IP literals."""
    with pytest.raises(SecurityValidationError) as exc:
        verify_deep_view_network_safety(prohibited_target, check_dns=False)
    assert "strictly prohibited" in str(exc.value) or "Unauthorized host" in str(exc.value)


def test_verify_deep_view_network_safety_rejects_dns_rebinding(monkeypatch):
    """Verify that DNS resolution to loopback or private network IPs is blocked."""
    import socket

    # Simulate malicious DNS resolving www.thedeepview.com to 127.0.0.1
    def mock_getaddrinfo(host, port, proto=0):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 443))]

    monkeypatch.setattr(socket, "getaddrinfo", mock_getaddrinfo)

    with pytest.raises(SecurityValidationError) as exc:
        verify_deep_view_network_safety("https://www.thedeepview.com/articles/test", check_dns=True)
    assert "prohibited IP '127.0.0.1'" in str(exc.value)


def test_fetch_rejects_disallowed_content_type():
    """Verify that fetch_with_strict_redirects rejects non-HTML/XML MIME types."""
    def binary_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=b"PK\x03\x04...",
            headers={"Content-Type": "application/zip"},
        )

    client = httpx.Client(transport=httpx.MockTransport(binary_handler))
    with pytest.raises(SecurityValidationError) as exc:
        fetch_with_strict_redirects("https://www.thedeepview.com/archive.zip", client=client)
    assert "Disallowed content type 'application/zip'" in str(exc.value)


def test_fetch_rejects_oversized_content_length():
    """Verify that responses declaring Content-Length > MAX_PAGE_BYTES are rejected immediately."""
    def oversized_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=b"<html>Hello</html>",
            headers={
                "Content-Type": "text/html",
                "Content-Length": str(MAX_PAGE_BYTES + 5000),
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(oversized_handler))
    with pytest.raises(SecurityValidationError) as exc:
        fetch_with_strict_redirects("https://www.thedeepview.com/large", client=client)
    assert "Content-Length" in str(exc.value) and "exceeds maximum allowed size" in str(exc.value)


def test_fetch_rejects_oversized_body():
    """Verify that responses with body size > MAX_PAGE_BYTES are rejected."""
    oversized_data = b"<p>text</p>" * (MAX_PAGE_BYTES // 10 + 100)

    def oversized_body_handler(request: httpx.Request) -> httpx.Response:
        resp = httpx.Response(
            200,
            stream=httpx.ByteStream(oversized_data),
            headers={"Content-Type": "text/html"},
        )
        resp.headers.pop("content-length", None)
        return resp

    client = httpx.Client(transport=httpx.MockTransport(oversized_body_handler))
    with pytest.raises(SecurityValidationError) as exc:
        fetch_with_strict_redirects("https://www.thedeepview.com/huge-page", client=client)
    assert "Response size" in str(exc.value) and "exceeds maximum allowed size" in str(exc.value)

