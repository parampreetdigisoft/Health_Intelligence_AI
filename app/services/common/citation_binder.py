"""
Attach verified source URLs to chat citations.

GPT may emit [source_N] or OpenAI Web Search tokens like [turn0search4].
The backend is the only place that inserts hrefs, using URLs from RAG
metadata or Web Search annotations — never model-guessed links.
"""

from __future__ import annotations

import re
from typing import Dict, List, Sequence, Set
from urllib.parse import urlparse

from app.services.common.url_verifier import is_valid_source_url

_SOURCE_ID_RE = re.compile(r"\[source_(\d+)\]", re.IGNORECASE)
_LABELED_SOURCE_RE = re.compile(
    r"\[([^\[\]]{1,120})\]\[source_(\d+)\]",
    re.IGNORECASE,
)
_MD_LINK_RE = re.compile(r"\[([^\[\]]{1,160})\]\((https?://[^)\s]+)\)")
_CATALOG_LINE_RE = re.compile(
    r"^source_(\d+)\s*\|\s*(.*?)\s*\|\s*(.*?)\s*\|\s*(https?://\S+)\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_TURN_ID_RE = re.compile(r"turn\d+(?:search|news|image)\d+", re.IGNORECASE)
_TURN_TOKEN_RE = re.compile(
    r"\[(turn\d+(?:search|news|image)\d+)\]"
    r"|【(?:\d+†)?(turn\d+(?:search|news|image)\d+)】",
    re.IGNORECASE,
)
_LABELED_TURN_RE = re.compile(
    r"\[([^\[\]]{1,160})\]((?:\[turn\d+(?:search|news|image)\d+\])+)",
    re.IGNORECASE,
)
_ADJACENT_LABEL_LINK_RE = re.compile(
    r"\[([^\[\]]{1,160})\]\[([^\[\]]{1,160})\]\((https?://[^)\s]+)\)"
)

_PUBLISHER_HINTS = (
    ("washington post", "washingtonpost.com"),
    ("nytimes", "nytimes.com"),
    ("new york times", "nytimes.com"),
    ("reuters", "reuters.com"),
    ("bbc", "bbc."),
    ("ap", "apnews.com"),
    ("associated press", "apnews.com"),
    ("hrw", "hrw.org"),
    ("human rights watch", "hrw.org"),
    ("amnesty", "amnesty.org"),
    ("marad", "maritime.dot.gov"),
    ("ocha", "unocha.org"),
    ("reliefweb", "reliefweb.int"),
    ("icg", "crisisgroup.org"),
    ("crisis group", "crisisgroup.org"),
    ("al jazeera", "aljazeera.com"),
    ("guardian", "theguardian.com"),
)

def _link_markdown(label: str, url: str) -> str:
    return f"[{label}]({url})"

def _source_label(source: Dict[str, str]) -> str:
    publisher = (source.get("publisher") or "").strip()
    title = (source.get("title") or "").strip()
    date = (source.get("date") or "").strip()
    if publisher and date:
        return f"{publisher}, {date}"
    if publisher:
        return publisher
    if title:
        return title[:80]
    return source.get("citation_id") or source.get("source_id") or "source"

def _host(url: str) -> str:
    host = urlparse(url).netloc.lower()
    return host[4:] if host.startswith("www.") else host

def _index_sources(sources: Sequence[Dict[str, str]]) -> Dict[str, Dict[str, str]]:
    indexed: Dict[str, Dict[str, str]] = {}
    search_i = 0
    news_i = 0
    for i, source in enumerate(sources, 1):
        sid = (source.get("source_id") or f"source_{i}").lower()
        indexed[sid] = source
        indexed[f"source_{i}"] = source
        cite = (source.get("citation_id") or "").lower()
        if cite:
            indexed[cite] = source
        # Fallback if the model used sequential OpenAI search tokens.
        if "news" in cite:
            indexed.setdefault(f"turn0news{news_i}", source)
            news_i += 1
        else:
            indexed.setdefault(f"turn0search{search_i}", source)
            indexed.setdefault(f"turn0search{i - 1}", source)
            search_i += 1
    return indexed

def _source_for_token(
    token: str,
    by_id: Dict[str, Dict[str, str]],
) -> Dict[str, str] | None:
    return by_id.get((token or "").lower())

def _source_for_label(
    label: str,
    sources: Sequence[Dict[str, str]],
    used: Set[str],
) -> Dict[str, str] | None:
    pub = (label.split(",")[0] if label else "").strip().lower()
    if not pub:
        return None
    for source in sources:
        url = source.get("url") or ""
        if url in used or not is_valid_source_url(url):
            continue
        hay = f"{source.get('title', '')} {url} {source.get('publisher', '')}".lower()
        if pub in hay:
            return source
        for name, host in _PUBLISHER_HINTS:
            if name in pub and host in hay:
                return source
            if name in pub and host in _host(url):
                return source
    return None

def apply_verified_citations(
    answer: str,
    sources: Sequence[Dict[str, str]],
) -> str:
    """
    Replace [source_N], [label][source_N], and [turn0searchN] with markdown
    links using verified URLs. Drop invented hrefs and leftover tool tokens.
    """
    if not answer:
        return answer

    verified = [s for s in sources if is_valid_source_url(s.get("url"))]
    by_id = _index_sources(verified)
    allowed_urls = {s["url"] for s in verified}
    used: Set[str] = set()

    def consume(source: Dict[str, str] | None) -> str:
        if not source:
            return ""
        url = source.get("url") or ""
        if not is_valid_source_url(url):
            return ""
        used.add(url)
        return url

    def replace_labeled_turns(match: re.Match[str]) -> str:
        label, token_blob = match.group(1), match.group(2)
        tokens = _TURN_ID_RE.findall(token_blob)
        links: List[str] = []
        first_url = ""
        for i, token in enumerate(tokens):
            source = _source_for_token(token, by_id) or (
                _source_for_label(label, verified, used) if i == 0 else None
            )
            url = consume(source)
            if not url:
                continue
            if i == 0:
                first_url = url
                links.append(_link_markdown(label, url))
            else:
                links.append(_link_markdown(_source_label(source or {}), url))
        if links:
            return " ".join(links)
        return f"[{label}]"

    rewritten = _LABELED_TURN_RE.sub(replace_labeled_turns, answer)

    def replace_turn_token(match: re.Match[str]) -> str:
        token = match.group(1) or match.group(2) or ""
        source = _source_for_token(token, by_id)
        url = consume(source)
        if not url:
            return ""
        return _link_markdown(_source_label(source or {}), url)

    rewritten = _TURN_TOKEN_RE.sub(replace_turn_token, rewritten)

    def replace_labeled_source(match: re.Match[str]) -> str:
        label, num = match.group(1), match.group(2)
        source = by_id.get(f"source_{num}")
        url = consume(source)
        if url:
            return _link_markdown(label, url)
        return f"[{label}]"

    rewritten = _LABELED_SOURCE_RE.sub(replace_labeled_source, rewritten)

    def replace_source_id(match: re.Match[str]) -> str:
        source = by_id.get(f"source_{match.group(1)}")
        url = consume(source)
        if not url:
            return ""
        return _link_markdown(_source_label(source or {}), url)

    rewritten = _SOURCE_ID_RE.sub(replace_source_id, rewritten)

    rewritten = _ADJACENT_LABEL_LINK_RE.sub(
        lambda m: _link_markdown(m.group(1), m.group(3))
        if m.group(3) in allowed_urls
        else f"[{m.group(1)}]",
        rewritten,
    )

    def replace_md(match: re.Match[str]) -> str:
        label, url = match.group(1), match.group(2)
        if url in allowed_urls or url in used:
            return match.group(0)
        return f"[{label}]"

    rewritten = _MD_LINK_RE.sub(replace_md, rewritten)
    rewritten = re.sub(r"\s{2,}", " ", rewritten)
    rewritten = re.sub(r" \.", ".", rewritten)
    return rewritten.strip()


def merge_verified_sources(
    web_sources: Sequence[Dict[str, str]],
) -> List[Dict[str, str]]:
    """RAG first, then web search; keep complete URLs; re-id sequentially."""
    merged: List[Dict[str, str]] = []
    seen: Set[str] = set()
    for source in list(web_sources):
        url = (source.get("url") or "").strip()
        if not is_valid_source_url(url) or url in seen:
            continue
        seen.add(url)
        item = dict(source)
        item["source_id"] = f"source_{len(merged) + 1}"
        item["url"] = url
        merged.append(item)
    return merged
