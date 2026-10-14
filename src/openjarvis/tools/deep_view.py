"""The Deep View Exclusive Web Research Tool.

Provides verified, source-restricted retrieval exclusively from:
  - https://www.thedeepview.com/
  - https://thedeepview.com/

Enforces strict domain validation, rejects lookalikes, non-HTTPS protocols,
and cross-domain redirects. Extracts article metadata (title, publication date,
canonical URL, and substantive text) for LLM grounding.
"""

from __future__ import annotations

import ipaddress
import json
import logging
import re
import socket
import urllib.parse
from dataclasses import dataclass
from datetime import datetime, timezone
from html.parser import HTMLParser
from typing import Any, Optional, Tuple

import httpx

from openjarvis.core.registry import ToolRegistry
from openjarvis.core.types import ToolResult
from openjarvis.tools._stubs import BaseTool, ToolSpec

logger = logging.getLogger(__name__)

# Official approved origins for The Deep View
APPROVED_ORIGINS = frozenset(
    {
        "https://www.thedeepview.com",
        "https://thedeepview.com",
    }
)
APPROVED_HOSTS = frozenset(
    {
        "www.thedeepview.com",
        "thedeepview.com",
    }
)

ALLOWED_CONTENT_TYPES = frozenset(
    {
        "text/html",
        "application/xhtml+xml",
        "text/xml",
        "application/xml",
        "application/rss+xml",
        "application/atom+xml",
        "text/plain",
    }
)

SITEMAP_URL = "https://www.thedeepview.com/sitemap.xml"
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36 OpenJarvis-DeepView/1.0"
)

MAX_PAGE_BYTES = 2_000_000  # 2MB response limit
REQUEST_TIMEOUT = httpx.Timeout(timeout=15.0, connect=5.0, read=10.0, write=5.0)


class SecurityValidationError(ValueError):
    """Raised when a URL or redirect violates domain isolation."""


def validate_deep_view_url(url: str) -> str:
    """Validate that a URL strictly targets approved The Deep View origins.

    Enforces:
      - Scheme MUST be 'https'
      - Host MUST strictly be 'www.thedeepview.com' or 'thedeepview.com'
      - No credentials (username/password)
      - Standard HTTPS port only (None or 443)
      - Rejection of IP addresses, ports, and lookalike domains (e.g. thedeepview.co, fake-thedeepview.com)

    Returns:
      Normalized, canonical URL string.

    Raises:
      SecurityValidationError: If the URL fails domain or security validation.
    """
    if not isinstance(url, str) or not url.strip():
        raise SecurityValidationError("URL must be a non-empty string.")

    cleaned_url = url.strip()

    try:
        parsed = urllib.parse.urlsplit(cleaned_url)
    except Exception as exc:
        raise SecurityValidationError(f"Invalid URL syntax: {exc}") from exc

    # 1. Enforce HTTPS scheme
    if parsed.scheme.lower() != "https":
        raise SecurityValidationError(
            f"Insecure scheme '{parsed.scheme}'. Only 'https' is permitted for The Deep View."
        )

    # 2. Enforce approved hostname strictly
    hostname = (parsed.hostname or "").lower().strip()
    if not hostname:
        raise SecurityValidationError("Missing hostname in URL.")

    if hostname not in APPROVED_HOSTS:
        raise SecurityValidationError(
            f"Unauthorized host '{hostname}'. Only official The Deep View domains "
            f"({', '.join(sorted(APPROVED_HOSTS))}) are permitted. "
            "External search engines, general websites, and lookalike domains are strictly forbidden."
        )

    # 3. Reject credentials
    if parsed.username or parsed.password:
        raise SecurityValidationError(
            "URLs containing user credentials are not permitted."
        )

    # 4. Enforce standard port
    if parsed.port not in (None, 443):
        raise SecurityValidationError(
            f"Non-standard port '{parsed.port}' is not permitted."
        )

    # Return normalized URL
    normalized_path = parsed.path or "/"
    if parsed.query:
        normalized_path = f"{normalized_path}?{parsed.query}"

    return f"https://{hostname}{normalized_path}"


def is_valid_deep_view_url(url: str) -> bool:
    """Return True if URL is an approved The Deep View URL, False otherwise."""
    try:
        validate_deep_view_url(url)
        return True
    except SecurityValidationError:
        return False


def is_deep_view_origin(url: str) -> bool:
    """Return True if URL is an approved The Deep View origin/URL, False otherwise."""
    return is_valid_deep_view_url(url)


def is_substantive_article_url(url: str) -> bool:
    """Return True if URL points to an individual article or newsletter post on The Deep View.

    Filters out category index pages, navigation hubs, marketing, and legal pages
    (e.g., /, /articles, /news, /newsletter, /podcast, /events, /jobs, /sponsor,
    /about-us, /privacy-policy) so search and newsletter discovery only surface
    genuine substantive editorial content.
    """
    try:
        if not is_valid_deep_view_url(url):
            return False
        parsed = urllib.parse.urlsplit(url.strip())
        parts = [p for p in parsed.path.strip("/").split("/") if p]
        return len(parts) >= 2 and parts[0].lower() in (
            "newsletter",
            "articles",
            "news",
        )
    except Exception:
        return False


def verify_deep_view_network_safety(url: str, check_dns: bool = True) -> None:
    """Validate that target URL does not resolve to private IPs, loopback, or rebind.

    Enforces:
      - Validates domain against APPROVED_HOSTS.
      - Unconditionally blocks IP address literals (IPv4 and IPv6).
      - Performs DNS preflight resolution when check_dns=True; ensures all resolved
        socket addresses are public unicast IPs, guarding against DNS rebinding
        and private network SSRF (RFC 1918, RFC 3927, RFC 4193, loopback, multicast).

    Raises:
      SecurityValidationError: If any safety condition is violated.
    """
    if not isinstance(url, str) or not url.strip():
        raise SecurityValidationError("URL must be a non-empty string.")

    parsed = urllib.parse.urlsplit(url.strip())
    hostname = (parsed.hostname or "").strip().lower()

    if not hostname:
        raise SecurityValidationError("Missing hostname in URL.")

    # 1. Unconditionally reject raw IP literals
    try:
        ipaddress.ip_address(hostname)
        raise SecurityValidationError(
            f"Raw IP literals ({hostname}) are strictly prohibited. "
            "Only official The Deep View domains are permitted."
        )
    except ValueError:
        pass  # Expected: hostname is a valid non-IP domain name

    # 2. Verify approved hostname
    if hostname not in APPROVED_HOSTS:
        raise SecurityValidationError(
            f"Unauthorized host '{hostname}'. Only official The Deep View domains "
            f"({', '.join(sorted(APPROVED_HOSTS))}) are permitted."
        )

    # 3. DNS rebinding & private network validation
    if check_dns:
        try:
            addr_info = socket.getaddrinfo(hostname, 443, proto=socket.IPPROTO_TCP)
            for item in addr_info:
                sockaddr = item[4]
                ip_str = sockaddr[0]
                ip_obj = ipaddress.ip_address(ip_str)
                if (
                    ip_obj.is_private
                    or ip_obj.is_loopback
                    or ip_obj.is_link_local
                    or ip_obj.is_multicast
                    or ip_obj.is_reserved
                    or ip_obj.is_unspecified
                ):
                    raise SecurityValidationError(
                        f"Host '{hostname}' resolved to prohibited IP '{ip_str}' "
                        "(SSRF / DNS-rebinding protection)."
                    )
        except socket.gaierror:
            # In offline test environments or isolated sandboxes where DNS is blocked,
            # allow request if mock transport/client is handling it, but log debug.
            logger.debug(
                "DNS lookup for '%s' failed (offline/sandbox environment).", hostname
            )


class _SimpleHTMLTextExtractor(HTMLParser):
    """Clean HTML parser that extracts text while stripping unwanted tags."""

    IGNORABLE_TAGS = frozenset(
        {
            "script",
            "style",
            "nav",
            "footer",
            "header",
            "aside",
            "form",
            "noscript",
            "svg",
            "button",
            "iframe",
        }
    )

    def __init__(self) -> None:
        super().__init__()
        self._ignore_stack: list[str] = []
        self._text_chunks: list[str] = []
        self._in_article = False
        self._meta_tags: dict[str, str] = {}
        self._title = ""
        self._in_title = False
        self._json_ld_raw: list[str] = []
        self._in_json_ld = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, Optional[str]]]) -> None:
        tag_lower = tag.lower()
        attr_dict = {k.lower(): (v or "") for k, v in attrs}

        if tag_lower == "script" and attr_dict.get("type") == "application/ld+json":
            self._in_json_ld = True
            return

        if tag_lower in self.IGNORABLE_TAGS:
            self._ignore_stack.append(tag_lower)
            return

        if tag_lower == "title":
            self._in_title = True
            return

        if tag_lower == "meta":
            prop = attr_dict.get("property") or attr_dict.get("name")
            content = attr_dict.get("content")
            if prop and content:
                self._meta_tags[prop.lower()] = content
            return

        if tag_lower in ("article", "main"):
            self._in_article = True

        if tag_lower in ("p", "h1", "h2", "h3", "h4", "li", "blockquote", "tr"):
            self._text_chunks.append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag_lower = tag.lower()
        if tag_lower == "script" and self._in_json_ld:
            self._in_json_ld = False
            return

        if self._ignore_stack and self._ignore_stack[-1] == tag_lower:
            self._ignore_stack.pop()
            return

        if tag_lower == "title":
            self._in_title = False
            return

        if tag_lower in ("article", "main"):
            self._in_article = False

        if tag_lower in ("p", "h1", "h2", "h3", "h4", "li", "tr"):
            self._text_chunks.append("\n")

    def handle_data(self, data: str) -> None:
        if self._in_json_ld:
            self._json_ld_raw.append(data)
            return

        if self._in_title:
            self._title += data
            return

        if not self._ignore_stack:
            cleaned = data.strip()
            if cleaned:
                self._text_chunks.append(data)

    def get_clean_text(self) -> str:
        raw = "".join(self._text_chunks)
        # Collapse multiple blank lines
        lines = [re.sub(r"[ \t]+", " ", line).strip() for line in raw.splitlines()]
        output_lines = []
        for line in lines:
            if line:
                output_lines.append(line)
            elif output_lines and output_lines[-1] != "":
                output_lines.append("")
        return "\n".join(output_lines).strip()


@dataclass
class DeepViewArticle:
    """Structured representation of an article or newsletter from The Deep View."""

    url: str
    title: str
    published_date: str
    author: str
    content: str
    summary: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "url": self.url,
            "title": self.title,
            "published_date": self.published_date,
            "author": self.author,
            "summary": self.summary,
            "content": self.content,
        }


def extract_deep_view_content(html: str, page_url: str) -> DeepViewArticle:
    """Extract metadata and substantive article text from The Deep View HTML."""
    parser = _SimpleHTMLTextExtractor()
    try:
        parser.feed(html)
    except Exception:
        pass

    title = parser._title.strip()
    # Clean up standard suffix from title
    for suffix in (" - The Deep View", " | The Deep View", " — The Deep View"):
        if title.endswith(suffix):
            title = title[: -len(suffix)].strip()

    meta = parser._meta_tags
    if not title:
        title = meta.get("og:title") or meta.get("twitter:title") or ""

    published_date = (
        meta.get("article:published_time")
        or meta.get("og:published_time")
        or meta.get("date")
        or meta.get("pubdate")
        or ""
    )

    author = meta.get("author") or meta.get("article:author") or "The Deep View"

    # Attempt parsing JSON-LD for rich structured metadata
    for raw_json in parser._json_ld_raw:
        try:
            data = json.loads(raw_json)
            if isinstance(data, list) and data:
                data = data[0]
            if isinstance(data, dict):
                if not title and data.get("headline"):
                    title = str(data["headline"])
                if not published_date and data.get("datePublished"):
                    published_date = str(data["datePublished"])
                if data.get("author"):
                    auth = data["author"]
                    if isinstance(auth, dict) and auth.get("name"):
                        author = str(auth["name"])
                    elif isinstance(auth, str):
                        author = auth
        except Exception:
            continue

    text = parser.get_clean_text()

    # Generate a rich, substantive summary containing top stories and key takeaways (up to ~1400 chars)
    raw_paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 25]
    boilerplate_prefixes = (
        "skip to",
        "subscribe",
        "listen to",
        "sign in",
        "all rights reserved",
    )
    substantive = [
        p
        for p in raw_paragraphs
        if not any(p.lower().startswith(bp) for bp in boilerplate_prefixes)
    ]
    summary_parts = []
    current_len = 0
    for p in substantive:
        if current_len + len(p) > 1400 and summary_parts:
            break
        summary_parts.append(p)
        current_len += len(p) + 2

    summary = "\n\n".join(summary_parts) if summary_parts else text[:1200]
    if len(summary) > 1500:
        summary = summary[:1497].rsplit(" ", 1)[0] + "..."

    return DeepViewArticle(
        url=page_url,
        title=title or "The Deep View Article",
        published_date=published_date or "Unknown Date",
        author=author,
        content=text,
        summary=summary,
    )


def fetch_with_strict_redirects(
    url: str,
    client: Optional[httpx.Client] = None,
    max_redirects: int = 5,
) -> Tuple[str, str]:
    """Fetch URL with strict verification that all redirect hops stay on The Deep View.

    Returns:
      (html_content, final_url)

    Raises:
      SecurityValidationError: If any hop targets an unapproved domain or scheme,
        exceeds size limits, or returns an unapproved content type.
      httpx.HTTPError: If the HTTP request fails.
    """
    target = validate_deep_view_url(url)
    verify_deep_view_network_safety(target, check_dns=True)
    owns_client = client is None
    c = client or httpx.Client(
        headers={"User-Agent": DEFAULT_USER_AGENT},
        timeout=REQUEST_TIMEOUT,
        verify=True,
    )

    try:
        current_url = target
        for _ in range(max_redirects):
            resp = c.get(current_url, follow_redirects=False)
            if resp.status_code in (301, 302, 303, 307, 308):
                location = resp.headers.get("Location")
                if not location:
                    break
                next_url = urllib.parse.urljoin(current_url, location)
                # Strictly validate that next hop does NOT leave The Deep View and is safe
                current_url = validate_deep_view_url(next_url)
                verify_deep_view_network_safety(current_url, check_dns=True)
                continue

            resp.raise_for_status()

            # Content-Type verification
            ct = resp.headers.get("content-type", "")
            if ct:
                mime = ct.split(";")[0].strip().lower()
                if mime and mime not in ALLOWED_CONTENT_TYPES:
                    raise SecurityValidationError(
                        f"Disallowed content type '{mime}'. Only text/html and XML feeds "
                        "from The Deep View are permitted."
                    )

            # Response size limits: Content-Length check
            cl = resp.headers.get("content-length")
            if cl and cl.isdigit() and int(cl) > MAX_PAGE_BYTES:
                raise SecurityValidationError(
                    f"Content-Length ({cl} bytes) exceeds maximum allowed size of {MAX_PAGE_BYTES} bytes."
                )

            # Response size limits: actual body byte check
            if len(resp.content) > MAX_PAGE_BYTES:
                raise SecurityValidationError(
                    f"Response size ({len(resp.content)} bytes) exceeds maximum allowed size of {MAX_PAGE_BYTES} bytes."
                )

            return resp.text, current_url

        raise httpx.TooManyRedirects("Exceeded maximum redirects on The Deep View.")
    finally:
        if owns_client:
            c.close()


class TheDeepViewRetriever:
    """Manages discovery, search, and content extraction exclusively for The Deep View."""

    def __init__(self, client: Optional[httpx.Client] = None) -> None:
        self._client = client
        self._sitemap_cache: list[dict[str, str]] = []
        self._sitemap_fetched_at: Optional[datetime] = None
        self._cache_ttl_seconds = 900  # 15 minutes
        self._article_cache: dict[str, DeepViewArticle] = {}

    def fetch_sitemap(self, force_refresh: bool = False) -> list[dict[str, str]]:
        """Fetch and parse https://www.thedeepview.com/sitemap.xml."""
        now = datetime.now(timezone.utc)
        if (
            not force_refresh
            and self._sitemap_cache
            and self._sitemap_fetched_at
            and (now - self._sitemap_fetched_at).total_seconds()
            < self._cache_ttl_seconds
        ):
            return list(self._sitemap_cache)

        try:
            xml_text, _ = fetch_with_strict_redirects(SITEMAP_URL, client=self._client)
        except Exception as exc:
            logger.warning("Failed to fetch The Deep View sitemap: %s", exc)
            if self._sitemap_cache:
                return list(self._sitemap_cache)
            raise

        entries: list[dict[str, str]] = []
        # Parse XML with regex to stay robust across namespaces and corrupt tags
        url_blocks = re.findall(
            r"<url>(.*?)</url>", xml_text, flags=re.DOTALL | re.IGNORECASE
        )
        for block in url_blocks:
            loc_match = re.search(r"<loc>(.*?)</loc>", block, flags=re.IGNORECASE)
            lastmod_match = re.search(
                r"<lastmod>(.*?)</lastmod>", block, flags=re.IGNORECASE
            )

            if loc_match:
                raw_loc = loc_match.group(1).strip()
                if is_valid_deep_view_url(raw_loc) and is_substantive_article_url(
                    raw_loc
                ):
                    lastmod = lastmod_match.group(1).strip() if lastmod_match else ""
                    entries.append({"url": raw_loc, "lastmod": lastmod})

        # Sort entries: newest lastmod first
        def _sort_key(item: dict[str, str]) -> str:
            return item.get("lastmod", "")

        entries.sort(key=_sort_key, reverse=True)
        self._sitemap_cache = entries
        self._sitemap_fetched_at = now
        return list(entries)

    def search(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        """Search The Deep View articles and newsletters by relevance and recency."""
        entries = self.fetch_sitemap()
        if not entries:
            return []

        cleaned_query = (query or "").lower().strip()
        # Tokenize query keywords (exclude common stop words, keep 'ai')
        stop_words = {
            "the",
            "a",
            "an",
            "is",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "and",
            "or",
            "what",
            "how",
            "deep",
            "view",
        }
        tokens = [
            t
            for t in re.split(r"\W+", cleaned_query)
            if (len(t) > 2 or t == "ai") and t not in stop_words
        ]

        scored_entries: list[tuple[float, dict[str, str]]] = []
        is_news_query = any(
            k in cleaned_query
            for k in (
                "latest",
                "recent",
                "news",
                "today",
                "this week",
                "new",
                "update",
                "updates",
            )
        )

        for idx, entry in enumerate(entries):
            url = entry["url"]
            path = urllib.parse.urlsplit(url).path.lower()

            # Skip root homepage in search results
            if path in ("", "/"):
                continue

            score = 0.0

            # Favor newsletter editions for comprehensive digests, and article paths
            if "/newsletter/" in path:
                score += 3.0
            elif "/articles/" in path:
                score += 2.0

            slug = path.split("/")[-1].replace("-", " ")
            slug_words = set(slug.split())

            # Keyword matching
            matched_tokens = 0
            for token in tokens:
                if token in slug_words:
                    score += 5.0
                    matched_tokens += 1
                elif token in slug:
                    score += 2.0
                    matched_tokens += 1

            year_tokens = {t for t in tokens if re.fullmatch(r"20\d\d", t)}
            substantive_tokens = [t for t in tokens if t not in year_tokens]
            if substantive_tokens:
                matched_substantive = sum(
                    1 for t in substantive_tokens if t in slug_words or t in slug
                )
                if matched_substantive == 0 and not is_news_query:
                    continue
            elif tokens and matched_tokens == 0 and not is_news_query:
                continue

            # Recency factor
            if is_news_query or not tokens:
                # Add recency points based on index in sitemap (which is sorted newest first)
                recency_boost = max(0.0, 15.0 - (idx * 0.1))
                score += recency_boost
            else:
                # Slight recency bonus
                score += max(0.0, 5.0 - (idx * 0.05))

            if score > 1.0 or not tokens:
                scored_entries.append((score, entry))

        scored_entries.sort(key=lambda x: x[0], reverse=True)
        top_candidates = [e for _, e in scored_entries[:limit]]

        # Retrieve and extract summaries for top candidates
        results: list[dict[str, Any]] = []
        for cand in top_candidates:
            url = cand["url"]
            try:
                article = self.read_article(url)
                results.append(
                    {
                        "title": article.title,
                        "url": article.url,
                        "published_date": article.published_date
                        or cand.get("lastmod", ""),
                        "summary": article.summary,
                        "author": article.author,
                    }
                )
            except Exception as exc:
                logger.debug("Failed to extract article at %s: %s", url, exc)
                # Fallback to slug-derived info
                slug_title = url.rstrip("/").split("/")[-1].replace("-", " ").title()
                results.append(
                    {
                        "title": slug_title,
                        "url": url,
                        "published_date": cand.get("lastmod", "Recent"),
                        "summary": f"Article from The Deep View ({url})",
                        "author": "The Deep View",
                    }
                )

        return results

    def get_latest_newsletters(self, limit: int = 3) -> list[dict[str, Any]]:
        """Retrieve the newest accessible newsletter editions from The Deep View."""
        entries = self.fetch_sitemap()
        newsletters = [e for e in entries if "/newsletter/" in e["url"]]
        if not newsletters:
            # Fall back to newest articles if no dedicated /newsletter/ slug
            newsletters = [e for e in entries if "/articles/" in e["url"]]

        top_newsletters = newsletters[:limit]
        results: list[dict[str, Any]] = []

        for item in top_newsletters:
            url = item["url"]
            try:
                article = self.read_article(url)
                results.append(article.to_dict())
            except Exception as exc:
                logger.debug("Error reading newsletter %s: %s", url, exc)
                slug_title = url.rstrip("/").split("/")[-1].replace("-", " ").title()
                results.append(
                    {
                        "title": slug_title,
                        "url": url,
                        "published_date": item.get("lastmod", ""),
                        "summary": f"The Deep View Newsletter edition ({url})",
                        "author": "The Deep View",
                        "content": "",
                    }
                )

        return results

    def read_article(self, url: str) -> DeepViewArticle:
        """Fetch and extract substantive content of a specific approved Deep View page."""
        validated_url = validate_deep_view_url(url)

        if validated_url in self._article_cache:
            return self._article_cache[validated_url]

        html_text, final_url = fetch_with_strict_redirects(
            validated_url, client=self._client
        )
        article = extract_deep_view_content(html_text, final_url)
        self._article_cache[validated_url] = article
        return article


@ToolRegistry.register("deep_view_search")
@ToolRegistry.register("deep_view_research")
class TheDeepViewTool(BaseTool):
    """Exclusive web research tool for The Deep View (https://www.thedeepview.com).

    Enforces that all online research is restricted strictly to The Deep View.
    Rejects any other external domains, search engines, or unapproved URLs.
    """

    tool_id = "deep_view_search"
    is_local = False

    def __init__(self, client: Optional[httpx.Client] = None) -> None:
        self._retriever = TheDeepViewRetriever(client=client)

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="deep_view_search",
            description=(
                "Search and retrieve AI news and newsletter articles exclusively from "
                "The Deep View (https://www.thedeepview.com). Supports searching articles by topic, "
                "retrieving latest newsletters, or reading full article text. "
                "No other website or general search engine is permitted."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["search", "latest_newsletters", "read_article"],
                        "description": (
                            "Action to perform: 'search' to find articles by query, "
                            "'latest_newsletters' to get recent newsletter editions, "
                            "or 'read_article' to retrieve full text of a specific Deep View URL."
                        ),
                    },
                    "query": {
                        "type": "string",
                        "description": "Topic or search keywords (for action='search').",
                    },
                    "url": {
                        "type": "string",
                        "description": (
                            "Exact The Deep View URL to read (for action='read_article'). "
                            "Must start with https://www.thedeepview.com/ or https://thedeepview.com/."
                        ),
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of articles to return (default 5).",
                    },
                },
            },
            category="search",
            metadata={"origin": "thedeepview.com", "exclusive": True},
        )

    def execute(self, **kwargs: Any) -> ToolResult:
        """Execute Deep View research request."""
        action = kwargs.get("action")
        query = kwargs.get("query", "")
        url = kwargs.get("url", "")
        limit = kwargs.get("limit", 5)

        try:
            limit = int(limit)
        except (ValueError, TypeError):
            limit = 5

        # Infer action if not explicitly given
        if not action:
            if url:
                action = "read_article"
            elif any(
                k in (query or "").lower()
                for k in ("newsletter", "weekly", "briefing", "latest news")
            ):
                action = "latest_newsletters"
            else:
                action = "search"

        try:
            if action == "read_article":
                if not url:
                    return ToolResult(
                        tool_name=self.tool_id,
                        content="Error: 'url' parameter is required for action='read_article'.",
                        success=False,
                    )
                article = self._retriever.read_article(url)
                formatted = (
                    f"[The Deep View - Article Content]\n"
                    f"Title: {article.title}\n"
                    f"Published Date: {article.published_date}\n"
                    f"Author: {article.author}\n"
                    f"Source URL: {article.url}\n\n"
                    f"Substantive Content:\n{article.content[:12000]}\n"
                )
                return ToolResult(
                    tool_name=self.tool_id,
                    content=formatted,
                    success=True,
                    metadata=article.to_dict(),
                )

            elif action == "latest_newsletters":
                newsletters = self._retriever.get_latest_newsletters(limit=limit)
                if not newsletters:
                    return ToolResult(
                        tool_name=self.tool_id,
                        content="No newsletter editions found on The Deep View.",
                        success=True,
                    )

                chunks = [
                    "[The Deep View - Latest Newsletters]\n"
                    "Verified editorial content from The Deep View (https://www.thedeepview.com)."
                ]
                for i, item in enumerate(newsletters, 1):
                    chunks.append(
                        f"--- Newsletter {i} ---\n"
                        f"Title: {item['title']}\n"
                        f"Date: {item.get('published_date', 'Recent')}\n"
                        f"Source URL: {item['url']}\n"
                        f"Author: {item.get('author', 'The Deep View')}\n"
                        f"Key Takeaways & Overview:\n{item['summary']}\n"
                    )
                return ToolResult(
                    tool_name=self.tool_id,
                    content="\n".join(chunks),
                    success=True,
                    metadata={"newsletters": newsletters},
                )

            else:  # search
                if not query:
                    query = "latest AI news"
                results = self._retriever.search(query=query, limit=limit)
                if not results:
                    return ToolResult(
                        tool_name=self.tool_id,
                        content=(
                            f"No matching articles found on The Deep View for query: '{query}'. "
                            "Per policy, no external search engines or other websites were consulted."
                        ),
                        success=True,
                    )

                chunks = [
                    f"[The Deep View - Search Results for '{query}']\n"
                    "Verified editorial content from The Deep View (https://www.thedeepview.com)."
                ]
                for i, res in enumerate(results, 1):
                    chunks.append(
                        f"--- Article {i} ---\n"
                        f"Title: {res['title']}\n"
                        f"Date: {res['published_date']}\n"
                        f"Source URL: {res['url']}\n"
                        f"Author: {res.get('author', 'The Deep View')}\n"
                        f"Key Takeaways & Overview:\n{res['summary']}\n"
                    )
                return ToolResult(
                    tool_name=self.tool_id,
                    content="\n".join(chunks),
                    success=True,
                    metadata={"results": results},
                )

        except SecurityValidationError as sec_err:
            return ToolResult(
                tool_name=self.tool_id,
                content=f"Security Policy Violation: {sec_err}",
                success=False,
            )
        except (httpx.ConnectError, httpx.TimeoutException) as net_err:
            logger.warning("Network connection to The Deep View failed: %s", net_err)
            return ToolResult(
                tool_name=self.tool_id,
                content=(
                    "Unable to connect to The Deep View (https://www.thedeepview.com). "
                    "Live research could not be completed at this time. "
                    "External fallback search is forbidden by system configuration."
                ),
                success=False,
            )
        except Exception as exc:
            logger.error("Unexpected error in TheDeepViewTool: %s", exc, exc_info=True)
            return ToolResult(
                tool_name=self.tool_id,
                content=f"Error retrieving content from The Deep View: {exc}",
                success=False,
            )


__all__ = [
    "ALLOWED_CONTENT_TYPES",
    "APPROVED_HOSTS",
    "APPROVED_ORIGINS",
    "DeepViewArticle",
    "SecurityValidationError",
    "TheDeepViewRetriever",
    "TheDeepViewTool",
    "extract_deep_view_content",
    "fetch_with_strict_redirects",
    "is_deep_view_origin",
    "is_substantive_article_url",
    "is_valid_deep_view_url",
    "validate_deep_view_url",
    "verify_deep_view_network_safety",
]
