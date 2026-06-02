"""
GDELT Doc API client with rate limiting and short-lived response cache.

GDELT requires at most one request every 5 seconds per client.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import httpx

logger = logging.getLogger(__name__)

# GDELT: "Please limit requests to one every 5 seconds"
GDELT_MIN_REQUEST_INTERVAL_SEC = 5.25
GDELT_RESPONSE_CACHE_TTL_SEC = 300
GDELT_MAX_RETRIES = 2

_lock = asyncio.Lock()
_last_request_at: float = 0.0
_cache: Dict[str, Tuple[float, List[Dict[str, Any]]]] = {}


def _is_rate_limited(resp: httpx.Response) -> bool:
    if resp.status_code == 429:
        return True
    try:
        text = (resp.text or "").lower()
    except Exception:
        return False
    return "limit requests" in text or "one every 5 seconds" in text


async def _wait_for_slot() -> None:
    """Ensure at least GDELT_MIN_REQUEST_INTERVAL_SEC between outbound GDELT calls."""
    global _last_request_at
    async with _lock:
        now = time.monotonic()
        elapsed = now - _last_request_at
        if elapsed < GDELT_MIN_REQUEST_INTERVAL_SEC:
            await asyncio.sleep(GDELT_MIN_REQUEST_INTERVAL_SEC - elapsed)
        _last_request_at = time.monotonic()


def _cache_get(cache_key: str) -> Optional[List[Dict[str, Any]]]:
    entry = _cache.get(cache_key)
    if not entry:
        return None
    ts, articles = entry
    if time.monotonic() - ts > GDELT_RESPONSE_CACHE_TTL_SEC:
        _cache.pop(cache_key, None)
        return None
    return articles


def _cache_set(cache_key: str, articles: List[Dict[str, Any]]) -> None:
    _cache[cache_key] = (time.monotonic(), articles)


async def fetch_doc_articles(url: str, cache_key: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    GET a GDELT doc API URL and return the articles list.

    Uses process-wide throttling, optional 5-minute cache, and limited retries on 429.
    """
    if cache_key:
        cached = _cache_get(cache_key)
        if cached is not None:
            logger.debug("GDELT cache hit for %s", cache_key)
            return cached

    last_error: Optional[Exception] = None

    for attempt in range(GDELT_MAX_RETRIES + 1):
        await _wait_for_slot()
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.get(url)

            if _is_rate_limited(resp):
                logger.warning(
                    "GDELT rate limit (attempt %s/%s); waiting %.1fs",
                    attempt + 1,
                    GDELT_MAX_RETRIES + 1,
                    GDELT_MIN_REQUEST_INTERVAL_SEC,
                )
                if attempt < GDELT_MAX_RETRIES:
                    await asyncio.sleep(GDELT_MIN_REQUEST_INTERVAL_SEC)
                    continue
                raise httpx.HTTPStatusError(
                    "GDELT rate limit exceeded",
                    request=resp.request,
                    response=resp,
                )

            resp.raise_for_status()

            try:
                payload = resp.json()
            except ValueError as exc:
                if _is_rate_limited(resp):
                    last_error = exc
                    if attempt < GDELT_MAX_RETRIES:
                        await asyncio.sleep(GDELT_MIN_REQUEST_INTERVAL_SEC)
                        continue
                raise

            articles = payload.get("articles") if isinstance(payload, dict) else None
            if not isinstance(articles, list):
                articles = []

            if cache_key and articles:
                _cache_set(cache_key, articles)

            return articles

        except httpx.HTTPStatusError as exc:
            last_error = exc
            if exc.response is not None and _is_rate_limited(exc.response):
                if attempt < GDELT_MAX_RETRIES:
                    await asyncio.sleep(GDELT_MIN_REQUEST_INTERVAL_SEC)
                    continue
            raise
        except httpx.HTTPError as exc:
            last_error = exc
            logger.warning("GDELT HTTP error (attempt %s): %s", attempt + 1, exc)
            if attempt < GDELT_MAX_RETRIES:
                await asyncio.sleep(GDELT_MIN_REQUEST_INTERVAL_SEC)
                continue
            raise

    if last_error:
        raise last_error
    return []
