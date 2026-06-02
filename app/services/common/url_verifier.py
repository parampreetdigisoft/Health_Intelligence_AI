"""
Verify news source URLs and build safe fallback search links when articles 404.
"""

from __future__ import annotations

import logging
from urllib.parse import quote_plus, urlparse

import httpx

logger = logging.getLogger(__name__)

_DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml",
}


def build_google_news_search_url(query: str) -> str:
    """Public Google News search — stable fallback when article URLs are unavailable."""
    
    query = quote_plus(query.strip())

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
    city: str,
    country: str,
    title: str,
) -> str:
    """
    Keep the URL if it loads; otherwise return a Google News search fallback.
    """

    if await is_url_live(url):
        return url

    search_query = f"{city} {country} {title}".strip()
    fallback = build_google_news_search_url(search_query)

    logger.warning(
        "Replacing dead source URL for %s, %s: %s -> %s",
        city,
        country,
        url,
        fallback,
    )

    return fallback

async def ensure_live_source_url_pillars(
    url: str,
    city: str,
    title: str,
) -> str:
    """
    Keep the URL if it loads; otherwise return a Google News search for the story.
    """
    if await is_url_live(url):
        return url

    fallback = build_google_news_search_url_pillars(city, title)
    logger.warning(
        "Replacing dead source URL for %s: %s -> %s",
        city,
        url,
        fallback,
    )
    return fallback

def build_google_news_search_url_pillars(city: str, title: str) -> str:
    """Public Google News search — stable fallback when article URLs are unavailable."""
    query = quote_plus(f"{city} {title}".strip())
    return (
        f"https://news.google.com/search?q={query}"
        "&hl=en-US&gl=US&ceid=US:en"
    )