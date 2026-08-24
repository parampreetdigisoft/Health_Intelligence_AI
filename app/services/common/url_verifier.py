"""
Verify news source URLs and build safe fallback search links when articles 404.
"""

from __future__ import annotations

import logging
from urllib.parse import quote_plus, urlparse
from typing import Optional
import httpx

logger = logging.getLogger(__name__)
_SEARCH_HOSTS = (
    "google.",
    "bing.com",
    "duckduckgo.com",
    "search.yahoo.",
    "news.google.",
)

_GENERIC_PATHS = {"", "/", "/world", "/news", "/en", "/latest", "/home", "/index"}


def is_search_results_url(url: str) -> bool:
    try:
        parsed = urlparse(url)
    except Exception:
        return False
    host = (parsed.netloc or "").lower()
    path = (parsed.path or "").lower()
    query = (parsed.query or "").lower()
    if any(h in host for h in _SEARCH_HOSTS):
        return True
    if "/search" in path or "q=" in query:
        return True
    return False


def is_homepage_url(url: str) -> bool:
    try:
        parsed = urlparse(url)
    except Exception:
        return False
    path = (parsed.path or "").rstrip("/") or "/"
    if path in _GENERIC_PATHS or path.lower() in _GENERIC_PATHS:
        return True
    segments = [s for s in path.split("/") if s]
    return len(segments) < 2


def is_valid_source_url(url: Optional[str]) -> bool:
    """True only for an absolute http(s) article/report URL — not search pages."""
    if not url or not isinstance(url, str):
        return False
    value = url.strip()
    if not value or " " in value or "..." in value:
        return False
    if not value.startswith(("http://", "https://")):
        return False
    try:
        parsed = urlparse(value)
    except Exception:
        return False
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        return False
    if is_search_results_url(value):
        return False
    if is_homepage_url(value):
        return False
    return True


_DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml",
}


def build_google_news_search_url(country: str, title: str) -> str:
    """Public Google News search — stable fallback when article URLs are unavailable."""
    query = quote_plus(f"{country} {title}".strip())
    return (
        f"https://news.google.com/search?q={query}"
        "&hl=en-US&gl=US&ceid=US:en"
    )


async def is_url_live(url: str) -> bool:
    """Return True if the URL responds without 404/410."""
    if not url or not url.startswith(("http://", "https://")):
        return False

    try:
        async with httpx.AsyncClient(
            follow_redirects=True,
            timeout=10.0,
        ) as client:
            response = await client.head(url, headers=_DEFAULT_HEADERS)
            if response.status_code == 405:
                response = await client.get(url, headers=_DEFAULT_HEADERS)

            if response.status_code in (404, 410):
                return False
            if response.status_code < 400:
                return True
            # Some publishers block bots with 403 but pages exist in browsers.
            if response.status_code == 403:
                host = urlparse(url).netloc.lower()
                if any(
                    trusted in host
                    for trusted in (
                        "reuters.com",
                        "bbc.co",
                        "bbc.com",
                        "apnews.com",
                        "theguardian.com",
                        "aljazeera.com",
                    )
                ):
                    return True
            return False
    except httpx.HTTPError as exc:
        logger.debug("URL check failed for %s: %s", url, exc)
        return False


async def ensure_live_source_url(
    url: str,
    country: str,
    title: str,
) -> str:
    """
    Keep the URL if it loads; otherwise return a Google News search for the story.
    """
    if await is_url_live(url):
        return url

    fallback = build_google_news_search_url(country, title)
    logger.warning(
        "Replacing dead source URL for %s: %s -> %s",
        country,
        url,
        fallback,
    )
    return fallback
