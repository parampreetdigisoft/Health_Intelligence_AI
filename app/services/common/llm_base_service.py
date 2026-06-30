"""
    PEM LLM Base Service
    --------------------
    Injectable, stateful wrapper around the LLM factory.

    Responsibilities:
      - Lazy initialisation with retry logic
      - Generic chain execution (prompt → LLM → StrOutputParser)
      - Raw single-prompt invocation (for simple text responses, e.g. RAG routing)
      - Retry decorator used by all consumer services

    Usage:
        from app.services.common.llm_base_service import LLMBaseService

        class MyService:
            def __init__(self):
                self._llm_svc = LLMBaseService()

            async def do_something(self):
                result = await self._llm_svc.invoke_chain(prompt_template, variables)
"""

import asyncio
import logging
from app.config import settings
from typing import Any, Dict, List, Optional
from langchain_core.messages import SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from app.services.common.llm_factory import llm_factory
logger = logging.getLogger(__name__)


class LLMBaseService:
    """
    Shared LLM lifecycle and invocation helpers.

    Design principles
    -----------------
    - One instance per consumer service (inject via __init__, not as a singleton).
    - All public methods are async-safe and include retry logic.
    - No business logic — purely LLM mechanics.
    """

    DEFAULT_MAX_RETRIES: int = 3
    DEFAULT_RETRY_DELAY: float = 1.0   # seconds

    def __init__(
        self,
        max_retries: int = DEFAULT_MAX_RETRIES,
        retry_delay: float = DEFAULT_RETRY_DELAY,
    ) -> None:
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._llm = None
        self._initialized = False

    # ------------------------------------------------------------------ #
    #  Initialisation                                                      #
    # ------------------------------------------------------------------ #

    async def initialize(self) -> None:
        """Lazily create the LLM instance with retry logic."""
        if self._initialized:
            return

        for attempt in range(self.max_retries):
            try:
                self._llm = llm_factory.create_llm()
                self._initialized = True
                logger.info(
                    "✅ LLMBaseService initialized with provider: %s",
                    settings.LLM_PROVIDER,
                )
                return
            except Exception as exc:
                logger.error(
                    "LLM init attempt %d/%d failed: %s",
                    attempt + 1, self.max_retries, exc,
                )
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay)
                else:
                    raise RuntimeError(
                        f"LLMBaseService: failed to initialise after "
                        f"{self.max_retries} attempts. Last error: {exc}"
                    ) from exc

    async def ensure_initialized(self) -> None:
        """Call before any LLM operation."""
        if not self._initialized or self._llm is None:
            await self.initialize()

    @property
    def llm(self):
        """Direct access to the underlying LLM (post-initialisation)."""
        if self._llm is None:
            raise RuntimeError(
                "LLM has not been initialised. "
                "Await ensure_initialized() or initialize() first."
            )
        return self._llm

    # ------------------------------------------------------------------ #
    #  Chain invocation  (prompt template → LLM → string)                #
    # ------------------------------------------------------------------ #

    async def invoke_chain(
        self,
        system_prompt: str,
        user_template: str,
        variables: Dict[str, Any],
        *,
        label: str = "chain",          # used only in log messages
    ) -> str:
        """
        Build a two-message ChatPromptTemplate, pipe through the LLM, and
        return the raw string output.  Retries on transient failures.

        Args:
            system_prompt:  Full system prompt text (already rendered).
            user_template:  User message Jinja/LangChain template string.
                            Use {variable_name} placeholders.
            variables:      Dict of template variables.
            label:          Human-readable label for log messages.

        Returns:
            Raw LLM response string.

        Raises:
            Last exception after exhausting all retries.
        """
        await self.ensure_initialized()

        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=system_prompt),
                ("user", user_template),
            ]
        )
        chain = prompt | self._llm | StrOutputParser()

        last_exc: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                result: str = await chain.ainvoke(variables)
                if result and result.strip() not in ("", "{}"):
                    return result
                logger.warning(
                    "[%s] Attempt %d returned empty/null — retrying.",
                    label, attempt + 1,
                )
            except Exception as exc:
                last_exc = exc
                logger.error(
                    "[%s] Attempt %d failed: %s", label, attempt + 1, exc
                )
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay)

        raise last_exc or RuntimeError(f"[{label}] All retries exhausted.")

    # ------------------------------------------------------------------ #
    #  Raw single-prompt invocation  (no template, returns content str)  #
    # ------------------------------------------------------------------ #

    async def invoke_raw(
        self,
        prompt: str,
        *,
        label: str = "raw_invoke",
    ) -> str:
        """
        Send a single plain-string prompt directly to the LLM and return the
        content string.  Useful for routing/classification tasks that do not
        need a structured ChatPromptTemplate.

        Args:
            prompt:  The full prompt text.
            label:   Human-readable label for log messages.

        Returns:
            LLM content as a string.
        """
        await self.ensure_initialized()

        last_exc: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                response = await self._llm.ainvoke(prompt)
                content = (
                    response.content
                    if hasattr(response, "content")
                    else str(response)
                )
                return content
            except Exception as exc:
                last_exc = exc
                logger.error(
                    "[%s] Raw invoke attempt %d failed: %s",
                    label, attempt + 1, exc,
                )
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay)

        raise last_exc or RuntimeError(f"[{label}] All raw invoke retries exhausted.")

    # ------------------------------------------------------------------ #
    #  Multi-turn message invocation  (for chat-style RAG answers)       #
    # ------------------------------------------------------------------ #

    async def invoke_messages(
        self,
        messages: List[Dict[str, str]],
        *,
        label: str = "messages_invoke",
    ) -> str:
        """
        Send a pre-built list of {role, content} dicts to the LLM and return
        the content string.  Used for chat-style multi-turn calls (e.g. RAG
        answer synthesis).

        Args:
            messages:  List of {"role": "system"|"user"|"assistant",
                                 "content": "..."}
            label:     Human-readable label for log messages.

        Returns:
            LLM content as a string.
        """
        await self.ensure_initialized()

        last_exc: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                response = await self._llm.ainvoke(messages)
                content = (
                    response.content
                    if hasattr(response, "content")
                    else str(response)
                )
                return content
            except Exception as exc:
                last_exc = exc
                logger.error(
                    "[%s] Messages invoke attempt %d failed: %s",
                    label, attempt + 1, exc,
                )
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay)

        raise last_exc or RuntimeError(
            f"[{label}] All message invoke retries exhausted."
        )

    # ------------------------------------------------------------------ #
    #  Utility                                                           #
    # ------------------------------------------------------------------ #

    def calculate_discrepancy(
        self,
        ai_progress: float,
        evaluator_score: Optional[float],
    ) -> float:
        """Return absolute difference between AI progress and evaluator score."""
        if evaluator_score is not None:
            return abs(ai_progress - evaluator_score)
        return ai_progress