"""
OpenAI Responses API web search for verified source URLs.

Used only when chat needs current/external verification. URLs come from
search-result annotations — never from model guesses.
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from app.config import settings, LLMProvider
from app.services.common.url_verifier import is_valid_source_url

logger = logging.getLogger(__name__)

_CACHE_TTL_SEC = 300
_cache: Dict[str, Tuple[float, str, List[Dict[str, str]]]] = {}
_TURN_IN_TEXT = re.compile(r"turn\d+(?:search|news|image)\d+", re.IGNORECASE)


def web_search_available() -> bool:
    """Web Search requires the native OpenAI provider and an API key."""
    provider = (settings.LLM_PROVIDER or "").lower()
    return provider == LLMProvider.OPENAI.value and bool(settings.OPENAI_API_KEY)


def _cache_get(key: str) -> Optional[Tuple[str, List[Dict[str, str]]]]:
    entry = _cache.get(key)
    if not entry:
        return None
    ts, text, sources = entry
    if time.monotonic() - ts > _CACHE_TTL_SEC:
        _cache.pop(key, None)
        return None
    return text, sources


def _cache_set(key: str, text: str, sources: List[Dict[str, str]]) -> None:
    _cache[key] = (time.monotonic(), text, sources)


def _attr_or_get(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _extract_text(response: Any) -> str:
    text = _attr_or_get(response, "output_text")
    if isinstance(text, str) and text.strip():
        return text

    parts: List[str] = []
    for item in _attr_or_get(response, "output", []) or []:
        for block in _attr_or_get(item, "content", []) or []:
            block_type = _attr_or_get(block, "type", "")
            value = _attr_or_get(block, "text")
            if block_type in ("output_text", "text") and isinstance(value, str):
                parts.append(value)
    return "\n".join(parts).strip()


def _add_source(
    sources: List[Dict[str, str]],
    seen: set[str],
    url: Any,
    title: Any = "",
    citation_id: Any = "",
) -> None:
    if not is_valid_source_url(str(url or "")):
        return
    clean = str(url).strip()
    cite = str(citation_id or "").strip().lower()
    for existing in sources:
        if existing["url"] == clean:
            if cite and not existing.get("citation_id"):
                existing["citation_id"] = cite
            if title and not existing.get("title"):
                existing["title"] = str(title).strip()
            return
    if clean in seen and not cite:
        return
    seen.add(clean)
    sources.append(
        {
            "source_id": f"source_{len(sources) + 1}",
            "citation_id": cite,
            "title": str(title or "").strip(),
            "url": clean,
            "publisher": "",
            "date": "",
        }
    )


def _citation_id_from_span(text: str, start: Any, end: Any) -> str:
    try:
        start_i = int(start)
        end_i = int(end)
    except (TypeError, ValueError):
        return ""
    if start_i < 0 or end_i > len(text) or start_i >= end_i:
        return ""
    snippet = text[start_i:end_i]
    match = _TURN_IN_TEXT.search(snippet)
    return match.group(0).lower() if match else ""


def _rewrite_annotation_spans(text: str, response: Any) -> str:
    """Replace annotated citation spans with markdown links using verified URLs."""
    spans: List[Tuple[int, int, str, str]] = []
    for item in _attr_or_get(response, "output", []) or []:
        for block in _attr_or_get(item, "content", []) or []:
            block_text = _attr_or_get(block, "text")
            if not isinstance(block_text, str) or block_text != text:
                continue
            for ann in _attr_or_get(block, "annotations", []) or []:
                url = _attr_or_get(ann, "url")
                if not is_valid_source_url(str(url or "")):
                    continue
                start = _attr_or_get(ann, "start_index")
                end = _attr_or_get(ann, "end_index")
                try:
                    start_i, end_i = int(start), int(end)
                except (TypeError, ValueError):
                    continue
                title = str(_attr_or_get(ann, "title") or "").strip()
                spans.append((start_i, end_i, str(url).strip(), title))

    if not spans:
        return text

    rewritten = text
    for start_i, end_i, url, title in sorted(spans, key=lambda s: s[0], reverse=True):
        if start_i < 0 or end_i > len(rewritten) or start_i >= end_i:
            continue
        snippet = rewritten[start_i:end_i]
        if "](" in snippet and snippet.strip().startswith("["):
            continue
        label = title or snippet.strip("[]【】")
        label = _TURN_IN_TEXT.sub("", label).strip() or title or "Source"
        rewritten = rewritten[:start_i] + f"[{label}]({url})" + rewritten[end_i:]
    return rewritten


def _extract_sources(response: Any, text: str) -> List[Dict[str, str]]:
    """Collect verified URLs from annotations, search actions, and results."""
    seen: set[str] = set()
    sources: List[Dict[str, str]] = []

    for item in _attr_or_get(response, "output", []) or []:
        item_type = str(_attr_or_get(item, "type") or "")

        if "web_search" in item_type:
            action = _attr_or_get(item, "action") or {}
            for src in _as_list(_attr_or_get(action, "sources")):
                _add_source(
                    sources,
                    seen,
                    _attr_or_get(src, "url") or _attr_or_get(src, "link"),
                    _attr_or_get(src, "title"),
                )
            for result in _as_list(_attr_or_get(item, "results")):
                _add_source(
                    sources,
                    seen,
                    _attr_or_get(result, "url") or _attr_or_get(result, "link"),
                    _attr_or_get(result, "title"),
                )

        for block in _attr_or_get(item, "content", []) or []:
            block_text = _attr_or_get(block, "text")
            ref_text = block_text if isinstance(block_text, str) else text
            for ann in _attr_or_get(block, "annotations", []) or []:
                ann_type = str(_attr_or_get(ann, "type") or "")
                if "url_citation" not in ann_type and ann_type not in ("citation", "url"):
                    continue
                cite = (
                    _attr_or_get(ann, "id")
                    or _attr_or_get(ann, "citation_id")
                    or _citation_id_from_span(
                        ref_text,
                        _attr_or_get(ann, "start_index"),
                        _attr_or_get(ann, "end_index"),
                    )
                )
                _add_source(
                    sources,
                    seen,
                    _attr_or_get(ann, "url"),
                    _attr_or_get(ann, "title"),
                    cite,
                )

    return sources


async def invoke_with_web_search(
    instructions: str,
    user_input: str,
    *,
    model: Optional[str] = None,
) -> Tuple[str, List[Dict[str, str]]]:
    """
    Call OpenAI Responses API with the built-in web_search tool.

    Returns (answer_text, verified_sources). Raises on failure so the
    caller can fall back to RAG-only generation.
    """
    if not web_search_available():
        raise RuntimeError("OpenAI Web Search is not configured")

    cache_key = hashlib.sha256(user_input.encode("utf-8", errors="ignore")).hexdigest()
    cached = _cache_get(cache_key)
    if cached:
        return cached

    try:
        from openai import AsyncOpenAI
    except ImportError as exc:
        raise RuntimeError("openai SDK is required for Web Search") from exc

    client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
    chosen_model = model or settings.OPENAI_MODEL
    last_error: Optional[Exception] = None

    for tool_type in ("web_search", "web_search_preview"):
        base_kwargs = {
            "model": chosen_model,
            "tools": [{"type": tool_type}],
            "instructions": instructions,
            "input": user_input,
            "temperature": settings.OPENAI_TEMPERATURE,
        }
        attempts = [
            {**base_kwargs, "include": ["web_search_call.action.sources"]},
            base_kwargs,
        ]
        for kwargs in attempts:
            try:
                response = await client.responses.create(**kwargs)
                text = _extract_text(response)
                if not text:
                    raise RuntimeError("Web Search returned empty text")
                sources = _extract_sources(response, text)
                text = _rewrite_annotation_spans(text, response)
                _cache_set(cache_key, text, sources)
                logger.info(
                    "Web Search completed via %s (%s verified URLs)",
                    tool_type,
                    len(sources),
                )
                return text, sources
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "Web Search with tool '%s' failed: %s",
                    tool_type,
                    exc,
                )

    raise last_error or RuntimeError("Web Search failed")
