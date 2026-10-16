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
from dataclasses import dataclass, field
from datetime import datetime, timezone
from html.parser import HTMLParser
from typing import Any, Optional, Tuple

import httpx

from openjarvis.core.registry import ToolRegistry
from openjarvis.core.types import ToolResult
from openjarvis.tools._stubs import BaseTool, ToolSpec
from openjarvis.tools.temporal import (
    CanonicalArticleRecord,
    deduplicate_articles,
    filter_and_rank_by_freshness,
    is_within_temporal_window,
    parse_iso_or_rfc_datetime,
)

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
FEED_URL = "https://www.thedeepview.com/feed.xml"
NEWSLETTER_ARCHIVE_URL = "https://www.thedeepview.com/newsletter"
ARTICLES_ARCHIVE_URL = "https://www.thedeepview.com/articles"

DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36 OpenJarvis-DeepView/1.0"
)

MAX_PAGE_BYTES = 5_000_000  # 5MB response limit (allows rich newsletter archive pages)
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
    published_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    retrieved_at: Optional[datetime] = None
    date_status: str = "unverified"
    raw_updated_date: str = ""
    editorial_date: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "url": self.url,
            "title": self.title,
            "published_date": self.published_date,
            "editorial_date": self.editorial_date,
            "published_at": self.published_at.isoformat() if self.published_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "retrieved_at": self.retrieved_at.isoformat() if self.retrieved_at else None,
            "date_status": self.date_status,
            "author": self.author,
            "summary": self.summary,
            "content": self.content,
        }

    def to_canonical(self) -> CanonicalArticleRecord:
        return CanonicalArticleRecord(
            url=self.url,
            title=self.title,
            author=self.author,
            published_at=self.published_at,
            updated_at=self.updated_at,
            retrieved_at=self.retrieved_at or datetime.now(timezone.utc),
            raw_published_date=self.published_date,
            raw_updated_date=self.raw_updated_date,
            editorial_date=self.editorial_date,
            date_status=self.date_status,
            content=self.content,
            summary=self.summary,
            source_name="The Deep View",
        )


def extract_deep_view_content(html: str, page_url: str) -> DeepViewArticle:
    """Extract metadata and substantive article text from The Deep View HTML."""
    retrieved_at = datetime.now(timezone.utc)
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
    updated_date = (
        meta.get("article:modified_time")
        or meta.get("og:updated_time")
        or meta.get("last-modified")
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
                if not updated_date and data.get("dateModified"):
                    updated_date = str(data["dateModified"])
                if data.get("author"):
                    auth = data["author"]
                    if isinstance(auth, dict) and auth.get("name"):
                        author = str(auth["name"])
                    elif isinstance(auth, str):
                        author = auth
        except Exception:
            continue

    published_at = parse_iso_or_rfc_datetime(published_date)
    updated_at = parse_iso_or_rfc_datetime(updated_date)

    # Extract human-visible editorial date from DOM (<time> tags, date paragraphs, bylines)
    editorial_date = ""

    # 1. Check <time> tag (prominently used on /articles/ pages)
    time_matches = re.findall(
        r"<time[^>]*>(.*?)</time>", html, flags=re.DOTALL | re.IGNORECASE
    )
    for tm in time_matches:
        d_m = re.search(r"([A-Za-z]+\s+\d{1,2},?\s+\d{4})", tm)
        if d_m:
            editorial_date = d_m.group(1).replace("  ", " ").strip()
            break

    # 2. Check newsletter date element (used on /newsletter/ pages)
    # e.g., <p class="text-gray-600 lg:text-xs text-xs">October 15, 2026</p>
    if not editorial_date:
        p_matches = re.findall(
            r'<p[^>]*class=["\'][^"\']*(?:text-gray|date|font|jsx)[^"\']*["\'][^>]*>(.*?)</p>',
            html,
            flags=re.IGNORECASE | re.DOTALL,
        )
        for p in p_matches:
            clean_p = re.sub(r"<[^>]+>", "", p).strip()
            d_m = re.match(
                r"^((?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},?\s+202\d)$",
                clean_p,
            )
            if d_m:
                editorial_date = d_m.group(1)
                break

    # 3. Fallback: byline date search in top section of document
    if not editorial_date:
        top_slice = html[:20000]
        d_m = re.search(
            r"\b((?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},?\s+202\d)\b",
            top_slice,
        )
        if d_m:
            editorial_date = d_m.group(1)

    # If published_at was not found via meta tags, attempt parsing from editorial_date
    if published_at is None and editorial_date:
        published_at = parse_iso_or_rfc_datetime(editorial_date)

    # If editorial_date was not extracted from DOM, derive from published_at
    if not editorial_date and published_at:
        if "/articles/" in page_url:
            editorial_date = published_at.strftime("%b %d, %Y")
        else:
            editorial_date = published_at.strftime("%B %d, %Y")

    if published_at is not None:
        date_status = "verified"
    elif published_date and published_date != "Unknown Date":
        date_status = "unverified"
    else:
        date_status = "missing"

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
        published_at=published_at,
        updated_at=updated_at,
        retrieved_at=retrieved_at,
        date_status=date_status,
        raw_updated_date=updated_date,
        editorial_date=editorial_date,
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
    """Manages discovery, search, and content extraction exclusively for The Deep View.

    Implements a multi-tier temporal retrieval strategy:
      - Tier 1 (Freshness & Breaking News): Official RSS 2.0 Feed (https://www.thedeepview.com/feed.xml) with verified RFC 2822 publication dates.
      - Tier 2 (Chronological Archive): Reverse-chronological newsletter editions from https://www.thedeepview.com/newsletter.
      - Tier 3 (Broad Topic Search): Sitemap-assisted search (https://www.thedeepview.com/sitemap.xml) with page-level publication verification.
    """

    def __init__(self, client: Optional[httpx.Client] = None) -> None:
        self._client = client
        self._sitemap_cache: list[dict[str, str]] = []
        self._sitemap_fetched_at: Optional[datetime] = None
        self._feed_cache: list[CanonicalArticleRecord] = []
        self._feed_fetched_at: Optional[datetime] = None
        self._newsletter_index_cache: list[str] = []
        self._newsletter_index_fetched_at: Optional[datetime] = None
        self._cache_ttl_seconds = 900  # 15 minutes for sitemap
        self._feed_ttl_seconds = 300  # 5 minutes for active feed
        self._article_cache: dict[str, DeepViewArticle] = {}

    def fetch_feed(self, force_refresh: bool = False) -> list[CanonicalArticleRecord]:
        """Fetch and parse https://www.thedeepview.com/feed.xml (RSS 2.0)."""
        now = datetime.now(timezone.utc)
        if (
            not force_refresh
            and self._feed_cache
            and self._feed_fetched_at
            and (now - self._feed_fetched_at).total_seconds() < self._feed_ttl_seconds
        ):
            return list(self._feed_cache)

        try:
            xml_text, _ = fetch_with_strict_redirects(FEED_URL, client=self._client)
        except Exception as exc:
            logger.debug("Failed to fetch The Deep View feed: %s", exc)
            if self._feed_cache:
                return list(self._feed_cache)
            return []

        records: list[CanonicalArticleRecord] = []
        item_blocks = re.findall(
            r"<item>(.*?)</item>", xml_text, flags=re.DOTALL | re.IGNORECASE
        )
        for item in item_blocks:
            title_m = re.search(
                r"<title>(?:<!\[CDATA\[)?(.*?)(?:\]\]>)?</title>",
                item,
                flags=re.IGNORECASE,
            )
            link_m = re.search(
                r"<link>(?:<!\[CDATA\[)?(.*?)(?:\]\]>)?</link>",
                item,
                flags=re.IGNORECASE,
            )
            pub_m = re.search(
                r"<pubDate>(?:<!\[CDATA\[)?(.*?)(?:\]\]>)?</pubDate>",
                item,
                flags=re.IGNORECASE,
            )
            author_m = re.search(
                r"<author>(?:<!\[CDATA\[)?(.*?)(?:\]\]>)?</author>",
                item,
                flags=re.IGNORECASE,
            )
            desc_m = re.search(
                r"<description>(?:<!\[CDATA\[)?(.*?)(?:\]\]>)?</description>",
                item,
                flags=re.DOTALL | re.IGNORECASE,
            )

            if not link_m:
                continue

            raw_url = link_m.group(1).strip()
            if not is_valid_deep_view_url(raw_url):
                continue

            title = title_m.group(1).strip() if title_m else "The Deep View Article"
            author = author_m.group(1).strip() if author_m else "The Deep View"
            raw_pub = pub_m.group(1).strip() if pub_m else ""
            pub_at = parse_iso_or_rfc_datetime(raw_pub)
            desc = desc_m.group(1).strip() if desc_m else ""
            clean_desc = re.sub(r"<[^>]+>", " ", desc).strip()
            editorial_date = pub_at.strftime("%b %d, %Y") if pub_at else ""

            record = CanonicalArticleRecord(
                url=raw_url,
                title=title,
                author=author,
                published_at=pub_at,
                updated_at=None,
                retrieved_at=now,
                raw_published_date=raw_pub,
                editorial_date=editorial_date,
                date_status="verified" if pub_at else "missing",
                content=clean_desc,
                summary=clean_desc[:1200],
                source_name="The Deep View",
            )
            records.append(record)

        # Sort newest published_at first
        records.sort(
            key=lambda r: (r.published_at.timestamp() if r.published_at else 0.0),
            reverse=True,
        )
        self._feed_cache = records
        self._feed_fetched_at = now
        return list(records)

    def fetch_newsletter_index(self, force_refresh: bool = False) -> list[str]:
        """Fetch chronological list of newsletter editions from https://www.thedeepview.com/newsletter."""
        now = datetime.now(timezone.utc)
        if (
            not force_refresh
            and self._newsletter_index_cache
            and self._newsletter_index_fetched_at
            and (now - self._newsletter_index_fetched_at).total_seconds()
            < self._cache_ttl_seconds
        ):
            return list(self._newsletter_index_cache)

        try:
            html_text, _ = fetch_with_strict_redirects(
                NEWSLETTER_ARCHIVE_URL, client=self._client
            )
        except Exception as exc:
            logger.debug("Failed to fetch newsletter archive: %s", exc)
            if self._newsletter_index_cache:
                return list(self._newsletter_index_cache)
            return []

        pattern = re.compile(r'href=["\'](/newsletter/[^"\'#?]+)["\']')
        links = pattern.findall(html_text)
        seen: set[str] = set()
        ordered: list[str] = []
        for l in links:
            if l not in seen and l != "/newsletter":
                full_url = f"https://www.thedeepview.com{l}"
                seen.add(l)
                ordered.append(full_url)

        self._newsletter_index_cache = ordered
        self._newsletter_index_fetched_at = now
        return list(ordered)

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

        self._sitemap_cache = entries
        self._sitemap_fetched_at = now
        return list(entries)

    def search(
        self,
        query: str,
        limit: int = 5,
        window: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """Search The Deep View articles and newsletters by relevance and verified recency."""
        cleaned_query = (query or "").lower().strip()
        stop_words = {
            "the", "a", "an", "is", "in", "on", "at", "to", "for",
            "of", "and", "or", "what", "how", "deep", "view",
        }
        tokens = [
            t
            for t in re.split(r"\W+", cleaned_query)
            if (len(t) > 2 or t == "ai") and t not in stop_words
        ]

        candidates: list[CanonicalArticleRecord] = []

        # 1. Search recent feed items first (highest freshness fidelity)
        try:
            feed_items = self.fetch_feed()
            for f in feed_items:
                title_lower = f.title.lower()
                desc_lower = f.content.lower()
                matched = any(t in title_lower or t in desc_lower for t in tokens)
                if matched or not tokens:
                    candidates.append(f)
        except Exception as exc:
            logger.debug("Feed search error: %s", exc)

        # 2. Search sitemap entries for broad keyword matching
        sitemap_entries = []
        try:
            sitemap_entries = self.fetch_sitemap()
        except Exception as exc:
            logger.debug("Sitemap search error: %s", exc)

        scored_sitemap: list[tuple[float, dict[str, str]]] = []
        is_news_query = any(
            k in cleaned_query
            for k in ("latest", "recent", "news", "today", "this week", "new", "update", "updates")
        )

        for entry in sitemap_entries:
            url = entry["url"]
            path = urllib.parse.urlsplit(url).path.lower()
            if path in ("", "/"):
                continue

            score = 0.0
            if "/newsletter/" in path:
                score += 3.0
            elif "/articles/" in path:
                score += 2.0

            slug = path.split("/")[-1].replace("-", " ")
            slug_words = set(slug.split())

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

            if score > 1.0 or not tokens:
                scored_sitemap.append((score, entry))

        scored_sitemap.sort(key=lambda x: x[0], reverse=True)

        # Inspect top sitemap candidate pages to extract genuine published_at
        for _, entry in scored_sitemap[: max(limit, 5)]:
            url = entry["url"]
            try:
                article = self.read_article(url)
                candidates.append(article.to_canonical())
            except Exception as exc:
                logger.debug("Failed reading candidate %s: %s", url, exc)

        deduped = deduplicate_articles(candidates)
        in_win, out_win = filter_and_rank_by_freshness(deduped, window=window)

        results_records = in_win[:limit] if in_win else out_win[:limit]
        results: list[dict[str, Any]] = []
        for r in results_records:
            results.append(
                {
                    "title": r.title,
                    "url": r.url,
                    "published_date": r.format_published_display(),
                    "editorial_date": r.editorial_date,
                    "published_at": r.published_at.isoformat() if r.published_at else None,
                    "updated_at": r.updated_at.isoformat() if r.updated_at else None,
                    "date_status": r.date_status,
                    "summary": r.summary,
                    "author": r.author,
                }
            )
        return results

    def get_latest_newsletters(
        self, limit: int = 5, window: str = "recent"
    ) -> list[dict[str, Any]]:
        """Retrieve the newest accessible newsletter editions in verified chronological order."""
        candidates: list[CanonicalArticleRecord] = []

        # Tier 1: Editorial reverse-chronological archive
        try:
            archive_urls = self.fetch_newsletter_index()
            for url in archive_urls[: max(limit * 2, 8)]:
                try:
                    art = self.read_article(url)
                    candidates.append(art.to_canonical())
                except Exception as exc:
                    logger.debug("Failed reading archive url %s: %s", url, exc)
        except Exception as exc:
            logger.debug("Newsletter index failed: %s", exc)

        # Tier 2: Check RSS feed ONLY if newsletter archive was empty or failed
        if not candidates:
            try:
                feed_items = self.fetch_feed()
                for f in feed_items:
                    candidates.append(f)
            except Exception as exc:
                logger.debug("Feed fetch in latest_newsletters failed: %s", exc)

        # Tier 3: Sitemap fallback if both index and feed were empty
        if not candidates:
            try:
                entries = self.fetch_sitemap()
                sitemap_newsletters = [
                    e["url"] for e in entries if "/newsletter/" in e["url"]
                ]
                if not sitemap_newsletters:
                    sitemap_newsletters = [
                        e["url"] for e in entries if "/articles/" in e["url"]
                    ]
                for url in sitemap_newsletters[:limit]:
                    try:
                        art = self.read_article(url)
                        candidates.append(art.to_canonical())
                    except Exception as exc:
                        logger.debug("Failed reading sitemap candidate %s: %s", url, exc)
            except Exception as exc:
                logger.debug("Sitemap fallback failed: %s", exc)

        deduped = deduplicate_articles(candidates)
        in_win, out_win = filter_and_rank_by_freshness(deduped, window=window)

        # If in_window contains items matching requested freshness, use them;
        # otherwise provide the latest available verified editorial editions.
        selected = in_win[:limit] if in_win else out_win[:limit]
        return [c.to_dict() for c in selected]

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
    Guarantees temporal provenance and verified publication dates.
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
                "retrieving latest newsletters with verified publication dates, or reading full article text. "
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
                    "time_window": {
                        "type": "string",
                        "enum": ["today", "yesterday", "last_24h", "recent", "any"],
                        "description": "Temporal freshness filter: 'today', 'last_24h', 'recent' (default), or 'any'.",
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
        window = kwargs.get("time_window") or kwargs.get("window", "recent")

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

        now_utc = datetime.now(timezone.utc)
        now_str = now_utc.strftime("%Y-%m-%d %H:%M UTC")

        try:
            if action == "read_article":
                if not url:
                    return ToolResult(
                        tool_name=self.tool_id,
                        content="Error: 'url' parameter is required for action='read_article'.",
                        success=False,
                    )
                article = self._retriever.read_article(url)
                if article.editorial_date:
                    pub_display = (
                        f"{article.editorial_date} (Verified Published Date)"
                        if article.date_status == "verified"
                        else f"{article.editorial_date} (Unverified)"
                    )
                elif article.published_at:
                    pub_display = (
                        f"{article.published_at.strftime('%Y-%m-%d %H:%M UTC')} (Verified Published Date)"
                        if article.date_status == "verified"
                        else f"{article.published_at.strftime('%Y-%m-%d %H:%M UTC')} (Unverified)"
                    )
                else:
                    pub_display = f"{article.published_date} (Unverified)"
                formatted = (
                    f"[The Deep View - Article Content]\n"
                    f"Title: {article.title}\n"
                    f"Publication Date: {pub_display}\n"
                    f"Date Status: {article.date_status.title()}\n"
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
                newsletters = self._retriever.get_latest_newsletters(
                    limit=limit, window=window
                )
                if not newsletters:
                    return ToolResult(
                        tool_name=self.tool_id,
                        content="No newsletter editions found on The Deep View for the requested window.",
                        success=True,
                    )

                chunks = [
                    f"[The Deep View - Latest Newsletters]\n"
                    f"Verified editorial content from The Deep View (https://www.thedeepview.com).\n"
                    f"Current Retrieval Time: {now_str}\n"
                    f"Temporal Grounding: Ground all dates strictly in the verified publication dates below."
                ]
                for i, item in enumerate(newsletters, 1):
                    editorial = item.get("editorial_date")
                    raw_pub = item.get("published_at")
                    dt_parsed = parse_iso_or_rfc_datetime(raw_pub) if raw_pub else None
                    status = item.get("date_status", "unverified").title()

                    if editorial and dt_parsed:
                        date_str = f"{editorial} ({dt_parsed.strftime('%Y-%m-%d %H:%M UTC')})"
                    elif editorial:
                        date_str = editorial
                    elif dt_parsed:
                        date_str = dt_parsed.strftime("%B %d, %Y (%Y-%m-%d %H:%M UTC)")
                    else:
                        date_str = item.get("published_date") or "Unknown / Unverified"

                    chunks.append(
                        f"--- Newsletter {i} ---\n"
                        f"Title: {item['title']}\n"
                        f"Publication Date: {date_str} ({status})\n"
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
                results = self._retriever.search(
                    query=query, limit=limit, window=window
                )
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
                    f"Verified editorial content from The Deep View (https://www.thedeepview.com).\n"
                    f"Current Retrieval Time: {now_str}\n"
                    f"Temporal Grounding: Ground all dates strictly in the verified publication dates below."
                ]
                for i, res in enumerate(results, 1):
                    date_disp = res.get("published_date") or "Unknown / Unverified"
                    chunks.append(
                        f"--- Article {i} ---\n"
                        f"Title: {res['title']}\n"
                        f"Publication Date: {date_disp}\n"
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
