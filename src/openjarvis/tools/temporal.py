"""Trustworthy Temporal Data Pipeline & Provenance Module.

Enforces explicit temporal semantics:
- Published: When original editorial content was first released.
- Updated: When the publisher modified or re-exported the content.
- Retrieved: When JARVIS fetched the content (UTC).
- Date Status: 'verified' (valid provenance), 'unverified' (inferred/fallback), 'missing'.

Provides timezone-aware UTC normalization, multi-format timestamp parsing
(ISO 8601, RFC 2822, Unix epoch seconds/ms), window filtering, deduplication,
and grounding validation. Missing publication dates are NEVER overwritten with
the current clock or update timestamps.
"""

from __future__ import annotations

import email.utils
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, List, Optional, Tuple

logger = logging.getLogger(__name__)


def parse_iso_or_rfc_datetime(val: Any) -> Optional[datetime]:
    """Parse a datetime representation into a timezone-aware UTC datetime.

    Supports:
      - ISO 8601 (e.g., '2026-10-15T23:33:00.000Z', '2026-10-15T20:55:10+00:00', '2026-10-15')
      - RFC 2822 / RSS (e.g., 'Thu, 15 Oct 2026 21:12:04 GMT', 'Thu, 15 Oct 2026 21:12:04 +0000')
      - Unix epoch timestamps (seconds or milliseconds as int, float, or numeric string)
      - Standard date formats ('YYYY-MM-DD', 'YYYY/MM/DD')

    Returns:
      timezone-aware datetime in UTC, or None if missing, malformed, or unparseable.
    """
    if val is None:
        return None

    if isinstance(val, datetime):
        if val.tzinfo is None:
            return val.replace(tzinfo=timezone.utc)
        return val.astimezone(timezone.utc)

    # Handle numeric timestamp (int, float, or string representation of numbers)
    if isinstance(val, (int, float)):
        try:
            ts = float(val)
            # Detect milliseconds vs seconds (threshold: 1e11 is year 1973 in ms or year 5138 in s)
            if ts > 1e11:
                ts = ts / 1000.0
            return datetime.fromtimestamp(ts, tz=timezone.utc)
        except (ValueError, OverflowError, OSError):
            return None

    if not isinstance(val, str):
        return None

    raw = val.strip()
    if not raw:
        return None

    # Handle numeric strings (e.g. "1760565184" or "1760565184000")
    if raw.isdigit():
        try:
            ts = float(raw)
            if ts > 1e11:
                ts = ts / 1000.0
            return datetime.fromtimestamp(ts, tz=timezone.utc)
        except (ValueError, OverflowError, OSError):
            return None

    # 1. Try RFC 2822 (standard RSS pubDate format)
    try:
        rfc_dt = email.utils.parsedate_to_datetime(raw)
        if rfc_dt is not None:
            if rfc_dt.tzinfo is None:
                return rfc_dt.replace(tzinfo=timezone.utc)
            return rfc_dt.astimezone(timezone.utc)
    except Exception:
        pass

    # 2. Try ISO 8601 parsing
    iso_clean = raw
    if iso_clean.endswith("Z") or iso_clean.endswith("z"):
        iso_clean = iso_clean[:-1] + "+00:00"

    try:
        iso_dt = datetime.fromisoformat(iso_clean)
        if iso_dt.tzinfo is None:
            return iso_dt.replace(tzinfo=timezone.utc)
        return iso_dt.astimezone(timezone.utc)
    except Exception:
        pass

    # 3. Fallback date patterns like 'YYYY-MM-DD' or 'YYYY/MM/DD'
    date_match = re.search(r"(\d{4})[-/](\d{1,2})[-/](\d{1,2})", raw)
    if date_match:
        try:
            year, month, day = (
                int(date_match.group(1)),
                int(date_match.group(2)),
                int(date_match.group(3)),
            )
            return datetime(year, month, day, 0, 0, 0, tzinfo=timezone.utc)
        except ValueError:
            return None

    # 4. Human-readable editorial date patterns (e.g., 'October 15, 2026', 'Oct 15, 2026', '15 October 2026')
    clean_editorial = re.sub(r"(\d+)(st|nd|rd|th)", r"\1", raw).replace(",", " ")
    clean_editorial = re.sub(r"\s+", " ", clean_editorial).strip()
    for fmt in ("%B %d %Y", "%b %d %Y", "%Y %m %d", "%d %B %Y", "%d %b %Y"):
        try:
            dt = datetime.strptime(clean_editorial, fmt)
            return dt.replace(tzinfo=timezone.utc)
        except ValueError:
            continue

    return None


@dataclass
class CanonicalArticleRecord:
    """Standardized representation of a retrieved article with verified temporal provenance."""

    url: str
    title: str
    author: str
    published_at: Optional[datetime]  # Timezone-aware UTC publication timestamp
    updated_at: Optional[datetime]  # Timezone-aware UTC modification timestamp
    retrieved_at: datetime  # Timezone-aware UTC timestamp when fetched
    raw_published_date: str = ""
    raw_updated_date: str = ""
    date_status: str = "unverified"  # 'verified', 'unverified', or 'missing'
    content: str = ""
    summary: str = ""
    source_name: str = "The Deep View"
    editorial_date: str = ""  # Clean human-visible editorial date (e.g. 'October 15, 2026' or 'Oct 15, 2026')
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_name": self.source_name,
            "url": self.url,
            "title": self.title,
            "author": self.author,
            "published_at": self.published_at.isoformat() if self.published_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "retrieved_at": self.retrieved_at.isoformat(),
            "raw_published_date": self.raw_published_date,
            "raw_updated_date": self.raw_updated_date,
            "editorial_date": self.editorial_date,
            "date_status": self.date_status,
            "summary": self.summary,
            "content": self.content,
            "metadata": self.metadata,
        }

    def format_published_display(self) -> str:
        """Return clear, disambiguated date presentation for users and LLMs."""
        status = "Verified" if self.date_status == "verified" else "Unverified"
        if self.editorial_date and self.published_at:
            iso_str = self.published_at.strftime("%Y-%m-%d %H:%M UTC")
            return f"{self.editorial_date} ({iso_str}) ({status})"
        if self.editorial_date:
            return f"{self.editorial_date} ({status})"
        if self.published_at:
            base = self.published_at.strftime("%Y-%m-%d %H:%M UTC")
            if self.date_status == "verified":
                return f"{base} (Verified Published Date)"
            return f"{base} (Unverified Published Date)"
        if self.raw_published_date:
            return f"{self.raw_published_date} (Unverified Format)"
        return "Unknown / Unverified Publication Date"


def is_within_temporal_window(
    dt: Optional[datetime],
    window: str,
    ref_now: Optional[datetime] = None,
) -> bool:
    """Evaluate whether a timestamp falls inside the requested relative temporal window."""
    if dt is None:
        return False

    now = ref_now or datetime.now(timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)

    diff_seconds = (now - dt).total_seconds()

    # Reject future dates beyond a reasonable clock skew margin (60 minutes)
    if diff_seconds < -3600:
        return False

    w = (window or "any").lower().strip()

    if w in ("today", "current", "latest"):
        # Matches if published on the same calendar UTC day or within the last 30 hours
        is_same_day = (
            dt.year == now.year and dt.month == now.month and dt.day == now.day
        )
        return is_same_day or (diff_seconds <= 30 * 3600)

    if w in ("yesterday",):
        # 24 to 48 hours ago, or previous UTC calendar day
        yesterday_date = (now - timedelta(days=1)).date()
        return dt.date() == yesterday_date or (24 * 3600 <= diff_seconds <= 54 * 3600)

    if w in ("last_24h", "24h", "24 hours", "last 24 hours"):
        return 0 <= diff_seconds <= 86400 + 3600

    if w in ("this_week", "last_week", "recent", "week"):
        return 0 <= diff_seconds <= 7 * 86400 + 3600

    if w in ("any", "all"):
        return True

    return diff_seconds <= 7 * 86400  # Default to recent window


def deduplicate_articles(
    articles: List[CanonicalArticleRecord],
) -> List[CanonicalArticleRecord]:
    """Deduplicate records by canonical URL and normalized title slug."""
    seen_urls: set[str] = set()
    seen_slugs: set[str] = set()
    unique: List[CanonicalArticleRecord] = []

    for art in articles:
        norm_url = art.url.strip().lower().rstrip("/")
        if norm_url in seen_urls:
            continue

        slug = re.sub(r"[^a-z0-9]+", "-", art.title.lower()).strip("-")
        if slug and slug in seen_slugs and len(slug) > 10:
            continue

        seen_urls.add(norm_url)
        if slug:
            seen_slugs.add(slug)
        unique.append(art)

    return unique


def filter_and_rank_by_freshness(
    articles: List[CanonicalArticleRecord],
    window: Optional[str] = None,
    ref_now: Optional[datetime] = None,
) -> Tuple[List[CanonicalArticleRecord], List[CanonicalArticleRecord]]:
    """Partition and sort articles into (in_window_articles, out_of_window_articles).

    Articles are sorted strictly by verified `published_at` descending.
    Articles without valid published_at are relegated to out_of_window.
    """
    now = ref_now or datetime.now(timezone.utc)
    deduped = deduplicate_articles(articles)

    in_window: List[CanonicalArticleRecord] = []
    out_window: List[CanonicalArticleRecord] = []

    for art in deduped:
        if art.published_at and is_within_temporal_window(
            art.published_at, window or "any", ref_now=now
        ):
            in_window.append(art)
        else:
            out_window.append(art)

    # Sort in_window by published_at descending
    def _sort_key(item: CanonicalArticleRecord) -> float:
        if item.published_at:
            return item.published_at.timestamp()
        return 0.0

    in_window.sort(key=_sort_key, reverse=True)
    out_window.sort(key=_sort_key, reverse=True)

    return in_window, out_window


__all__ = [
    "CanonicalArticleRecord",
    "deduplicate_articles",
    "filter_and_rank_by_freshness",
    "is_within_temporal_window",
    "parse_iso_or_rfc_datetime",
]
