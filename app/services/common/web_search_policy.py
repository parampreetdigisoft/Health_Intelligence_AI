"""Decide when chat should call OpenAI Web Search."""

from __future__ import annotations

import re

_AHI_ONLY = re.compile(
    r"\b(ahi score|pillar (score|rating)|kpi|index score|ahi assessment|"
    r"peace enablers matrix score)\b",
    re.I,
)
_CURRENT_INTEL = re.compile(
    r"\b(latest|current|recent|today|this week|this month|developments?|"
    r"conflict|war|risks?|humanitarian|security|crisis|escalat|"
    r"ceasefire|outbreak|situation|news|globally|worldwide)\b",
    re.I,
)
_NEEDS_SOURCE = re.compile(
    r"\b(when did|who (is|was|signed)|independence|source|according to|"
    r"what happened|cite)\b",
    re.I,
)


def _normalize_rag_context(rag_context) -> str:
    if rag_context is None:
        return ""
    if isinstance(rag_context, str):
        return rag_context
    if isinstance(rag_context, (list, tuple, set)):
        return "\n".join(str(item) for item in rag_context)
    if isinstance(rag_context, dict):
        return "\n".join(f"{key}: {value}" for key, value in rag_context.items())
    return str(rag_context)


def question_needs_web_search(question: str, rag_context: object) -> bool:
    """
    Web Search is conditional:
    - AHI scores / KPIs / pillar ratings → RAG only
    - current conflict/risk/humanitarian or source-needed facts → search
    - empty RAG for a non-AHI question → search
    """
    q = question or ""
    context_text = _normalize_rag_context(rag_context)

    if _AHI_ONLY.search(q) and not _CURRENT_INTEL.search(q):
        return False
    if _CURRENT_INTEL.search(q) or _NEEDS_SOURCE.search(q):
        return True
    if not context_text.strip():
        return True
    return False
