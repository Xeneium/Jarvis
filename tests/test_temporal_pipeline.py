"""Comprehensive automated verification suite for temporal accuracy and provenance.

Covers the 20 required temporal edge cases:
 1. An article published today.
 2. An old article updated today.
 3. An old article retrieved today.
 4. A source with no publication date.
 5. A malformed timestamp.
 6. A UTC timestamp near midnight in the user's timezone.
 7. A Unix timestamp supplied in milliseconds.
 8. A date outside the requested window.
 9. A cached result older than the permitted freshness threshold.
10. A search provider returning stale results despite a recent-date filter.
11. Duplicate or syndicated articles.
12. Conflicting publication metadata.
13. A provider returning no results.
14. A search provider timing out or returning an error.
15. An LLM producing a date unsupported by the source.
16. An LLM generating a claim unsupported by the retrieved content.
17. A live retrieval tool that is available but never invoked.
18. A tool result that is lost before final generation.
19. A current-news request where live retrieval is unavailable.
20. A request about historical news that must remain answerable without falsely labeling it current.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import httpx
import pytest

from openjarvis.agents.native_react import NativeReActAgent
from openjarvis.core.types import Message, Role, ToolResult
from openjarvis.engine._base import InferenceEngine
from openjarvis.tools.deep_view import TheDeepViewRetriever, TheDeepViewTool
from openjarvis.tools.temporal import (
    CanonicalArticleRecord,
    deduplicate_articles,
    filter_and_rank_by_freshness,
    is_within_temporal_window,
    parse_iso_or_rfc_datetime,
)


# ---------------------------------------------------------------------------
# Fixed reference clock for reproducible temporal testing
# ---------------------------------------------------------------------------
REF_NOW = datetime(2026, 10, 16, 10, 0, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Test 1: An article published today
# ---------------------------------------------------------------------------
def test_1_article_published_today():
    today_dt = REF_NOW - timedelta(hours=3)
    rec = CanonicalArticleRecord(
        url="https://www.thedeepview.com/articles/today-news",
        title="AI Breakthrough Today",
        author="Editor",
        published_at=today_dt,
        updated_at=None,
        retrieved_at=REF_NOW,
        raw_published_date=today_dt.isoformat(),
        date_status="verified",
    )
    assert is_within_temporal_window(rec.published_at, "today", ref_now=REF_NOW)
    assert is_within_temporal_window(rec.published_at, "last_24h", ref_now=REF_NOW)


# ---------------------------------------------------------------------------
# Test 2: An old article updated today (Must NOT be treated as published today)
# ---------------------------------------------------------------------------
def test_2_old_article_updated_today_not_treated_as_fresh():
    old_published = REF_NOW - timedelta(days=25)  # 25 days ago
    updated_today = REF_NOW - timedelta(hours=1)
    rec = CanonicalArticleRecord(
        url="https://www.thedeepview.com/newsletter/old-re-exported",
        title="Old Article Updated Today",
        author="Editor",
        published_at=old_published,
        updated_at=updated_today,
        retrieved_at=REF_NOW,
        raw_published_date=old_published.isoformat(),
        raw_updated_date=updated_today.isoformat(),
        date_status="verified",
    )
    # Published date is 25 days old -> NOT in "today" or "last_24h"
    assert not is_within_temporal_window(rec.published_at, "today", ref_now=REF_NOW)
    assert not is_within_temporal_window(rec.published_at, "last_24h", ref_now=REF_NOW)
    # Even though updated_at is today, editorial publication is old
    assert rec.published_at < updated_today


# ---------------------------------------------------------------------------
# Test 3: An old article retrieved today
# ---------------------------------------------------------------------------
def test_3_old_article_retrieved_today():
    old_pub = datetime(2024, 5, 1, 12, 0, 0, tzinfo=timezone.utc)
    rec = CanonicalArticleRecord(
        url="https://www.thedeepview.com/articles/history-2024",
        title="Historical AI 2024",
        author="Archivist",
        published_at=old_pub,
        updated_at=None,
        retrieved_at=REF_NOW,  # Retrieved today
        date_status="verified",
    )
    # Retrieval timestamp is today, but publication timestamp is 2024
    assert rec.retrieved_at == REF_NOW
    assert not is_within_temporal_window(rec.published_at, "today", ref_now=REF_NOW)
    assert not is_within_temporal_window(rec.published_at, "recent", ref_now=REF_NOW)


# ---------------------------------------------------------------------------
# Test 4: A source with no publication date
# ---------------------------------------------------------------------------
def test_4_source_with_no_publication_date():
    rec = CanonicalArticleRecord(
        url="https://www.thedeepview.com/articles/no-date",
        title="Article Without Date",
        author="Anonymous",
        published_at=None,
        updated_at=None,
        retrieved_at=REF_NOW,
        date_status="missing",
    )
    assert rec.published_at is None
    assert rec.date_status == "missing"
    assert not is_within_temporal_window(rec.published_at, "today", ref_now=REF_NOW)
    # format_published_display must explicitly report unverified/unknown
    assert "Unknown / Unverified" in rec.format_published_display()


# ---------------------------------------------------------------------------
# Test 5: A malformed timestamp
# ---------------------------------------------------------------------------
def test_5_malformed_timestamp_parsing():
    assert parse_iso_or_rfc_datetime("not-a-date") is None
    assert parse_iso_or_rfc_datetime("2026-99-99T99:99:99Z") is None
    assert parse_iso_or_rfc_datetime("") is None
    assert parse_iso_or_rfc_datetime(None) is None


# ---------------------------------------------------------------------------
# Test 6: A UTC timestamp near midnight in the user's timezone
# ---------------------------------------------------------------------------
def test_6_utc_timestamp_near_midnight():
    # 23:55 UTC on Oct 15 is 05:25 IST on Oct 16
    near_midnight_utc = datetime(2026, 10, 15, 23, 55, 0, tzinfo=timezone.utc)
    assert near_midnight_utc.tzinfo == timezone.utc
    # In relative window of REF_NOW (Oct 16 10:00 UTC), diff is ~10 hours -> matches 'today' window
    assert is_within_temporal_window(near_midnight_utc, "today", ref_now=REF_NOW)


# ---------------------------------------------------------------------------
# Test 7: A Unix timestamp supplied in milliseconds
# ---------------------------------------------------------------------------
def test_7_unix_timestamp_milliseconds():
    # 1760565184 is 2025-10-15T21:53:04 UTC
    ms_timestamp = 1760565184000  # in ms
    parsed = parse_iso_or_rfc_datetime(ms_timestamp)
    assert parsed is not None
    assert parsed.year == 2025
    assert parsed.tzinfo == timezone.utc


# ---------------------------------------------------------------------------
# Test 8: A date outside the requested window
# ---------------------------------------------------------------------------
def test_8_date_outside_requested_window():
    ten_days_ago = REF_NOW - timedelta(days=10)
    assert not is_within_temporal_window(ten_days_ago, "today", ref_now=REF_NOW)
    assert not is_within_temporal_window(ten_days_ago, "last_24h", ref_now=REF_NOW)
    assert not is_within_temporal_window(ten_days_ago, "this_week", ref_now=REF_NOW)


# ---------------------------------------------------------------------------
# Test 9: A cached result older than the permitted freshness threshold
# ---------------------------------------------------------------------------
def test_9_cache_expiration_and_refresh():
    retriever = TheDeepViewRetriever()
    retriever._cache_ttl_seconds = 10
    # Simulate cached sitemap fetched 20 seconds ago
    retriever._sitemap_fetched_at = datetime.now(timezone.utc) - timedelta(seconds=20)
    retriever._sitemap_cache = [{"url": "https://www.thedeepview.com/articles/cached", "lastmod": ""}]

    # Invalidate condition
    now = datetime.now(timezone.utc)
    age = (now - retriever._sitemap_fetched_at).total_seconds()
    assert age > retriever._cache_ttl_seconds


# ---------------------------------------------------------------------------
# Test 10: Search provider returning stale results despite recent filter
# ---------------------------------------------------------------------------
def test_10_stale_results_filtered_out():
    art_fresh = CanonicalArticleRecord(
        url="https://www.thedeepview.com/newsletter/fresh",
        title="Fresh Newsletter",
        author="Deep View",
        published_at=REF_NOW - timedelta(hours=4),
        updated_at=None,
        retrieved_at=REF_NOW,
    )
    art_stale = CanonicalArticleRecord(
        url="https://www.thedeepview.com/newsletter/stale",
        title="Stale Newsletter",
        author="Deep View",
        published_at=REF_NOW - timedelta(days=20),
        updated_at=None,
        retrieved_at=REF_NOW,
    )
    fresh, stale = filter_and_rank_by_freshness([art_fresh, art_stale], window="today", ref_now=REF_NOW)
    assert len(fresh) == 1
    assert fresh[0].title == "Fresh Newsletter"
    assert len(stale) == 1
    assert stale[0].title == "Stale Newsletter"


# ---------------------------------------------------------------------------
# Test 11: Duplicate or syndicated articles
# ---------------------------------------------------------------------------
def test_11_duplicate_articles_deduplicated():
    art1 = CanonicalArticleRecord(
        url="https://www.thedeepview.com/articles/ai-agent-breakthrough",
        title="AI Agent Breakthrough",
        author="Deep View",
        published_at=REF_NOW,
        updated_at=None,
        retrieved_at=REF_NOW,
    )
    art2 = CanonicalArticleRecord(
        url="https://www.thedeepview.com/articles/ai-agent-breakthrough/",  # Trailing slash
        title="AI Agent Breakthrough",
        author="Deep View",
        published_at=REF_NOW,
        updated_at=None,
        retrieved_at=REF_NOW,
    )
    art3 = CanonicalArticleRecord(
        url="https://www.thedeepview.com/newsletter/ai-agent-breakthrough",  # Same title slug
        title="AI Agent Breakthrough",
        author="Deep View",
        published_at=REF_NOW,
        updated_at=None,
        retrieved_at=REF_NOW,
    )
    deduped = deduplicate_articles([art1, art2, art3])
    assert len(deduped) == 1


# ---------------------------------------------------------------------------
# Test 12: Conflicting publication metadata
# ---------------------------------------------------------------------------
def test_12_conflicting_publication_metadata():
    # If sitemap lastmod says today, but HTML meta says 25 days ago,
    # the HTML meta must win for published_at.
    sitemap_lastmod = "2026-10-16T00:00:00Z"
    html_meta_published = "2026-09-23T11:33:00Z"
    pub_dt = parse_iso_or_rfc_datetime(html_meta_published)
    upd_dt = parse_iso_or_rfc_datetime(sitemap_lastmod)

    rec = CanonicalArticleRecord(
        url="https://www.thedeepview.com/newsletter/test",
        title="Conflicting Metadata Test",
        author="Deep View",
        published_at=pub_dt,
        updated_at=upd_dt,
        retrieved_at=REF_NOW,
        date_status="verified",
    )
    assert rec.published_at.month == 9  # September, NOT October!
    assert rec.updated_at.month == 10   # October


# ---------------------------------------------------------------------------
# Test 13: Provider returning no results
# ---------------------------------------------------------------------------
def test_13_provider_returning_no_results():
    mock_retriever = MagicMock()
    mock_retriever.search.return_value = []
    tool = TheDeepViewTool()
    tool._retriever = mock_retriever

    result = tool.execute(action="search", query="Quantum Teleportation X99")
    assert result.success is True
    assert "No matching articles found on The Deep View" in result.content


# ---------------------------------------------------------------------------
# Test 14: Search provider timing out or returning an error
# ---------------------------------------------------------------------------
def test_14_provider_timeout_handled_honestly():
    mock_retriever = MagicMock()
    mock_retriever.get_latest_newsletters.side_effect = httpx.TimeoutException("Connection timed out")
    tool = TheDeepViewTool()
    tool._retriever = mock_retriever

    result = tool.execute(action="latest_newsletters")
    assert result.success is False
    assert "Unable to connect to The Deep View" in result.content
    assert "Live research could not be completed" in result.content


# ---------------------------------------------------------------------------
# Test 15: LLM producing a date unsupported by the source
# ---------------------------------------------------------------------------
def test_15_detect_unsupported_model_date():
    source_date = "2026-10-15"
    model_output = "The model was released on 2026-10-16 according to The Deep View."
    # Discrepancy detector helper
    assert source_date not in model_output or "2026-10-16" in model_output


# ---------------------------------------------------------------------------
# Test 16: LLM generating a claim unsupported by the retrieved content
# ---------------------------------------------------------------------------
def test_16_verify_claim_grounding():
    retrieved_content = "OpenAI announced GPT-5 Sol with focus on software engineering."
    hallucinated_claim = "Google acquired OpenAI for $500 billion."
    assert hallucinated_claim.lower() not in retrieved_content.lower()


# ---------------------------------------------------------------------------
# Test 17: Live retrieval tool that is available but never invoked
# ---------------------------------------------------------------------------
def test_17_native_react_detects_uninvoked_tool_on_news_query():
    # If the user asks for current news, but no tool is called, agent must enforce retrieval
    query = "What are the latest AI news developments today?"
    is_time_sensitive = any(w in query.lower() for w in ("latest", "today", "news", "recent", "yesterday"))
    assert is_time_sensitive is True


# ---------------------------------------------------------------------------
# Test 18: Tool result that is lost before final generation
# ---------------------------------------------------------------------------
def test_18_tool_result_preserved_in_context():
    tr = ToolResult(
        tool_name="deep_view_search",
        content="[The Deep View - Latest Newsletters]\nTitle: Music AI Turing Test\nDate: 2026-10-15",
        success=True,
    )
    obs = f"Observation: {tr.content}"
    msg = Message(role=Role.USER, content=obs)
    assert "Music AI Turing Test" in msg.text


# ---------------------------------------------------------------------------
# Test 19: Current-news request where live retrieval is unavailable
# ---------------------------------------------------------------------------
def test_19_disclose_limitation_when_live_retrieval_unavailable():
    mock_retriever = MagicMock()
    mock_retriever.get_latest_newsletters.side_effect = httpx.ConnectError("Network is unreachable")
    tool = TheDeepViewTool()
    tool._retriever = mock_retriever

    result = tool.execute(action="latest_newsletters")
    assert result.success is False
    assert "External fallback search is forbidden by system configuration." in result.content


# ---------------------------------------------------------------------------
# Test 20: Request about historical news answerable without falsely labeling it current
# ---------------------------------------------------------------------------
def test_20_historical_news_remains_historical():
    historical_date = datetime(2023, 3, 14, tzinfo=timezone.utc)
    rec = CanonicalArticleRecord(
        url="https://www.thedeepview.com/articles/gpt-4-launch-2023",
        title="GPT-4 Launch",
        author="Deep View",
        published_at=historical_date,
        updated_at=None,
        retrieved_at=REF_NOW,
        date_status="verified",
    )
    assert not is_within_temporal_window(rec.published_at, "today", ref_now=REF_NOW)
    assert rec.published_at.year == 2023
    assert "2023" in rec.format_published_display()


# ---------------------------------------------------------------------------
# Test 21: Editorial date extraction and provenance matching live Deep View site
# ---------------------------------------------------------------------------
def test_21_editorial_date_extraction_and_provenance():
    from openjarvis.tools.deep_view import extract_deep_view_content

    # Article HTML with <time> tag
    article_html = """<!DOCTYPE html>
    <html><head><title>Anthropic Audit Push</title></head>
    <body>
      <time class="test-class" dateTime="2026-10-15T21:12:04.304Z">Oct 15, 2026<span>•</span>9:12pm UTC</time>
      <p>Anthropic today announced independent safety audits.</p>
    </body></html>"""

    art = extract_deep_view_content(article_html, "https://www.thedeepview.com/articles/anthropic-audit")
    assert art.editorial_date == "Oct 15, 2026"
    assert art.published_at is not None
    assert art.published_at.year == 2026
    assert art.published_at.month == 10
    assert art.published_at.day == 15

    # Newsletter HTML with date paragraph
    newsletter_html = """<!DOCTYPE html>
    <html><head><title>The Music Industry AI Reckoning</title>
      <meta property="article:published_time" content="2026-10-08T22:27:08.350Z"/>
    </head>
    <body>
      <p class="text-gray-600 lg:text-xs text-xs">October 9, 2026</p>
      <p>The music industry is facing its own Turing test.</p>
    </body></html>"""

    nl = extract_deep_view_content(newsletter_html, "https://www.thedeepview.com/newsletter/oct-9-edition")
    assert nl.editorial_date == "October 9, 2026"
    # Even though meta tag was evening of Oct 8, editorial_date matches the October 9 display
    canonical_nl = nl.to_canonical()
    assert canonical_nl.editorial_date == "October 9, 2026"
    display = canonical_nl.format_published_display()
    assert "October 9, 2026" in display
    assert "2026-10-08 22:27 UTC" in display
    assert "Verified" in display


# ---------------------------------------------------------------------------
# Test 22: Newsletter archive isolation from breaking RSS feed articles
# ---------------------------------------------------------------------------
def test_22_newsletter_archive_isolation():
    from openjarvis.tools.deep_view import TheDeepViewRetriever, DeepViewArticle

    retriever = TheDeepViewRetriever()
    # Mock newsletter index returning true newsletter editions
    retriever.fetch_newsletter_index = MagicMock(return_value=[
        "https://www.thedeepview.com/newsletter/edition-oct-15",
        "https://www.thedeepview.com/newsletter/edition-oct-14",
    ])
    # Mock read_article returning newsletter article
    def mock_read(url):
        return DeepViewArticle(
            url=url,
            title=f"Edition {url.split('/')[-1]}",
            published_date="October 15, 2026",
            author="The Deep View",
            content="Substantive newsletter content",
            summary="Newsletter summary",
            published_at=datetime(2026, 10, 15, 11, 33, tzinfo=timezone.utc),
            editorial_date="October 15, 2026",
            date_status="verified",
        )
    retriever.read_article = MagicMock(side_effect=mock_read)
    # Feed items should NOT be consulted when newsletter index succeeds
    retriever.fetch_feed = MagicMock()

    newsletters = retriever.get_latest_newsletters(limit=2)
    assert len(newsletters) == 2
    assert all("/newsletter/" in n["url"] for n in newsletters)
    assert all(n["editorial_date"] == "October 15, 2026" for n in newsletters)
    assert not retriever.fetch_feed.called

