# =========================================================================== #
#  rag_query_service.py  (refactored)                                         #
# =========================================================================== #
"""
RAGQueryService  (refactored)
------------------------------
Two-stage RAG pipeline for country document Q&A.

Stage 1 — LLM-driven TOC routing  (which sections are relevant?)
Stage 2 — ChromaDB vector search within those sections

LLM calls are handled by LLMBaseService.
All prompt text comes from AHIPromptTemplates.
"""

from datetime import datetime, timedelta, timezone

import os
import re
import chromadb
import logging
import json
import httpx
from typing import List, Dict, Any, Optional
from app.services.common.embedding import create_embedding_function
from app.services.common.llm_base_service import LLMBaseService
from app.services.common.country_prompt import AHIPromptTemplates
from app.services.common.gdelt_client import fetch_doc_articles
from app.services.common.pillar_prompts import AHIPPillarPrompts
from app.services.core.repository import DatabaseRepository
from app.services.common.citation_binder import (
    apply_verified_citations,
    merge_verified_sources
)
from app.services.common.web_search_client import (
    invoke_with_web_search,
    web_search_available,
)
from app.services.common.web_search_policy import question_needs_web_search
from app.services.common import json_response_parser as jrp
logger = logging.getLogger(__name__)

CHROMA_PATH = "./chroma_store"


class RAGQueryService:
    """
    Hybrid RAG service: LLM-routed TOC selection + ChromaDB vector retrieval.

    LLM mechanics live in LLMBaseService (injected).
    Prompt text lives in AHIPromptTemplates.
    """

    def __init__(self) -> None:

        # Ensure directory exists
        if not os.path.exists(CHROMA_PATH):
            os.makedirs(CHROMA_PATH)

        try:
            self.client = chromadb.PersistentClient(
                path=CHROMA_PATH,
                settings=chromadb.config.Settings(anonymized_telemetry=False),
            )

        except Exception as e:
            logger.error(f"ChromaDB initialization failed: {e}")
            raise

        self.embed_fn = create_embedding_function()
        self._db = DatabaseRepository()
        # --- LLM (shared base service) ---
        self._llm_svc = LLMBaseService(max_retries=3, retry_delay=1.0)

    # ------------------------------------------------------------------ #
    #  Initialisation                                                    #
    # ------------------------------------------------------------------ #

    async def initialize(self) -> None:
        """Initialise the shared LLM service."""
        await self._llm_svc.initialize()

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    async def get_country_document_context(
        self,
        country_id: int,
        msg_text: str,
        pillar_id: Optional[int] = None,
    ) -> str:
        """
        Answer a natural-language question about a country using:
          1. LLM-selected TOC sections
          2. ChromaDB vector search within those sections
          3. LLM synthesis of retrieved chunks + chat history
        """
        # Stage 1 — TOC routing
        toc = await self._get_country_toc(country_id, pillar_id)

        relevant_toc_ids = []
        if len(toc) > 4:
            relevant_toc_ids = await self._get_relevant_Id(msg_text, toc)
        else:
            relevant_toc_ids = [row["TOCID"] for row in toc]

        # Stage 2 — Vector retrieval
        chunks = self._fetch_relevant_chunks(
            question=msg_text,
            toc_ids=relevant_toc_ids,
            country_id=country_id,
            pillar_id=pillar_id,
            top_k=10,
        )

        # Build context and history strings
        local_context = self._build_context_block(chunks)

        return local_context

    async def get_global_document_context(self, msg_text: str) -> str:

        toc = await self._get_global_toc()

        relevant_toc_ids = []
        if len(toc) > 4:
            relevant_toc_ids = await self._get_relevant_Id(msg_text, toc)
        else:
            relevant_toc_ids = [row["TOCID"] for row in toc]

        # Stage 2 — Vector retrieval
        chunks = self._fetch_relevant_chunks(
            question=msg_text,
            toc_ids=relevant_toc_ids,
            country_id=None,
            pillar_id=None,
            top_k=10,
        )

        # Build context and history strings
        local_context = self._build_context_block(chunks)

        return local_context

    async def send_question_to_llm(
        self,
        questionText: str,
        ai_context: str,
        countryName: str,
        pillar_name: str,
        historyText: Optional[str] = None,
    ) -> str:

        # Stage 3 — LLM answer synthesis
        return await self._synthesize_chat_answer(
            questionText, ai_context, countryName, pillar_name, historyText
        )

    async def send_cross_comparision_question_to_llm(
        self,
        questionText: str,
        ai_context: str,
        countryName: str,
        pillar_name: str,
        historyText: Optional[str] = None,
    ) -> str:

         return await self._synthesize_chat_answer(
            questionText, ai_context, countryName, pillar_name, historyText
        )

    async def _synthesize_chat_answer(
        self,
        questionText: str,
        ai_context: str,
        countryName: str,
        pillar_name: str,
        historyText: Optional[str] = None,
    ) -> str:
        system_prompt = AHIPromptTemplates.chat_system_prompt()
        user_prompt = AHIPromptTemplates.chat_answer_user_prompt(
            ai_context, historyText, questionText, countryName, pillar_name
        )

        if question_needs_web_search(questionText, ai_context) and web_search_available():
            try:
                answer, web_sources = await invoke_with_web_search(
                    system_prompt,
                    user_prompt,
                    model=None,
                )
                return apply_verified_citations(
                    answer, merge_verified_sources(web_sources)
                )
            except Exception as exc:
                logger.warning(
                    "Web Search unavailable; falling back to RAG-only: %s", exc
                )

        answer = await self._llm_svc.invoke_messages(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            label=f"rag_answer|country{countryName}",
        )
     
        return apply_verified_citations(answer,[])
    
    
    # ------------------------------------------------------------------ #
    #  Stage 1 — DB: fetch TOC                                           #
    #  ⚡ Tenant migration point: only this method touches the DB        #
    # ------------------------------------------------------------------ #

    async def _get_country_toc(
        self,
        country_id: int,
        pillar_id: Optional[int] = None,
    ) -> List[Dict]:
        """
        Fetch the Table-of-Contents entries for a country's uploaded documents.

        Returns a list of dicts with keys:
            TOCID, SectionPath, SectionTitle, SectionLevel, PillarID, FileName
        """
        query = """
            SELECT t.TOCID, t.SectionPath, t.SectionTitle, t.SectionLevel,
                   t.PillarID, cd.FileName
            FROM DocumentTOC t
            JOIN CountryDocuments cd ON cd.CountryDocumentID = t.CountryDocumentID
            WHERE t.CountryID = ? AND cd.IsDeleted = 0
        """
        # Future: add   AND t.TenantID = ?   when multi-tenant
        return await self._db.engine.fetch_dicts_async(query, (country_id,))

    async def _get_global_toc(self) -> List[Dict]:

        query = """
            SELECT t.TOCID, t.SectionPath, t.SectionTitle, t.SectionLevel,
                   t.PillarID, cd.FileName
            FROM DocumentTOC t
            JOIN CountryDocuments cd ON cd.CountryDocumentID = t.CountryDocumentID
            WHERE  cd.IsDeleted = 0 or DocumentLevel Like ?
        """
        documentLevel = "Global"

        return await self._db.engine.fetch_dicts_async(query, (documentLevel))

    # ------------------------------------------------------------------ #
    #  Stage 1 — LLM: route question to relevant TOC sections            #
    # ------------------------------------------------------------------ #

    async def _get_relevant_Id(
        self,
        question: str,
        toc: List[Dict],
    ) -> List[int]:
        """
        Ask the LLM which TOC section IDs are most relevant to the question.
        Returns a list of TOCID integers (may be empty).
        """
        if not toc:
            return []

        toc_text = "\n".join(
            f"[{row['TOCID']}] (Level {row['SectionLevel']}) {row['SectionPath']}"
            for row in toc
        )
        prompt = AHIPromptTemplates.get_relevant_Id_prompt(toc_text, question)
        raw = await self._llm_svc.invoke_raw(
            prompt, label=f"rag_routing|q={question[:40]}"
        )

        match = re.search(r"\[[\d,\s]*\]", raw)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        return []

    # ------------------------------------------------------------------ #
    #  Stage 2 — ChromaDB: vector search within sections                 #
    # ------------------------------------------------------------------ #

    def _fetch_relevant_chunks(
        self,
        question: str,
        toc_ids: List[int],
        country_id: Optional[int] = None,
        pillar_id: Optional[int] = None,
        top_k: int = 5,
    ) -> List[Dict]:
        """
        Run a vector similarity search against the ChromaDB collection and
        return the top-k chunks, optionally filtered to the routed TOC IDs.
        """
        collection_name = (
            "Global"
            if country_id is None
            else (
                f"Country_{country_id}"
                if pillar_id is None
                else f"Country_Pillar_{country_id}"
            )
        )
        try:
            #    collections = self.client.list_collections()

            collection = self.client.get_collection(
                name=collection_name, embedding_function=self.embed_fn
            )
        except Exception as e:
            logger.error(f"Error fetching collection {collection_name}: {e}")
            return []

        where_filter = {"toc_id": {"$in": toc_ids}} if toc_ids else None
        results = collection.query(
            query_texts=[question],
            n_results=top_k,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )

        chunks = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            chunks.append(
                {
                    "text": doc,
                    "section": meta.get("section_path", ""),
                    "file": meta.get("section_title", ""),
                    "relevance": round(1 - dist, 3),
                     "source_url": meta.get("source_url") or meta.get("sourceUrl") or "",
                    "source_name": meta.get("source_name") or meta.get("sourceName") or "",
                    "published_date": meta.get("published_date") or meta.get("source_data_year") or "",
                }
            )
        return chunks

    async def get_related_FAQ_IDs(
        self,
        question: str,
        toc: List[Dict],
    ) -> List[int]:
        """
        Ask the LLM which FAQ section IDs are most relevant to the question.
        Returns a list of FAQIDs integers (may be empty).
        """
        if not toc:
            return []

        toc_text = "\n".join(
            f"[{row['FAQID']}] (QuestionText {row['QuestionText']}) {row['Category']}"
            for row in toc
        )
        prompt = AHIPromptTemplates.get_relevant_faqId_prompt(toc_text, question)
        raw = await self._llm_svc.invoke_raw(
            prompt, label=f"rag_routing|q={question[:80]}"
        )

        match = re.search(r"\[[\d,\s]*\]", raw)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        return []


    async def country_executive_slides( self,  country_name: str, ai_country_context: str, allPillarContexts: str, year: int = None) -> Dict[str, Any]:

        try:

            # ---------------------------------------------------------
            # SYSTEM PROMPT
            # ---------------------------------------------------------
            system_prompt = (
                AHIPromptTemplates.Country_executive_slides_prompt(
                    publicContext=ai_country_context,
                    allPillarContexts=allPillarContexts
                )
            )

            # ---------------------------------------------------------
            # USER TEMPLATE
            # ---------------------------------------------------------
            user_template = """
            country:
            {country_name}

            Year:
            {year}
            """

            # ---------------------------------------------------------
            # LLM CALL
            # ---------------------------------------------------------
            raw = await self._llm_svc.invoke_chain(
                system_prompt=system_prompt,
                user_template=user_template,
                variables={
                    "country_name": country_name,
                    "year": year
                },
                label=f"country-executive-slides|{country_name}",
            )

            analysis = json.loads(
                jrp.clean_json_response(raw)
            )

            return {
                "success": True,
                "data": analysis
            }

        except Exception as exc:
            logger.exception(
                "country_executive_slides failed"
            )

            return {
                "success": False,
                "error": str(exc)
            }


    async def _fetch_gdelt_emerging_articles(
        self,
        max_records: int,
        query_variant: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Fetch GDELT articles (one variant per request, 5s throttle between calls).
        Tries at most two variants if the first returns no articles.
        """
        countries = await self._db.get_active_countries()
        all_country_codes, region_groups = AHIPromptTemplates.build_gdelt_country_scope(
            countries
        )

        variant_count = AHIPromptTemplates.gdelt_emerging_variant_count()
        start_idx = (
            query_variant
            if query_variant is not None
            else AHIPromptTemplates.pick_gdelt_emerging_variant_index()
        ) % variant_count

        last_error: Optional[Exception] = None
        max_variant_tries = 2 if query_variant is None else 1

        for attempt in range(max_variant_tries):
            idx = (start_idx + attempt) % variant_count
            gdelt_url, _ = AHIPromptTemplates.emerging_trends_gdelt_url(
                max_records,
                all_country_codes,
                region_groups,
                variant_index=idx,
            )
            cache_key = f"emerging:{max_records}:{idx}"

            try:
                articles_raw = await fetch_doc_articles(gdelt_url, cache_key=cache_key)
                if articles_raw:
                    return articles_raw
                logger.warning(
                    "GDELT variant %s returned no articles",
                    idx,
                )
            except Exception as exc:
                last_error = exc
                logger.warning("GDELT fetch failed for variant %s: %s", idx, exc)
                if attempt + 1 >= max_variant_tries:
                    raise

        if last_error:
            raise last_error
        raise ValueError("GDELT returned no articles")

    async def emerging_trends_and_issues(
        self,
        country_count: int = 8,
        query_variant: Optional[int] = None,
    ) -> Dict[str, Any]:
        try:
            max_records = max(1, min(250, country_count))

            now_utc = datetime.now(timezone.utc)

            # ---------------------------------------------------------
            # Fetch articles from GDELT (last 24h, English; Africa health rotated query)
            # ---------------------------------------------------------
            articles_raw = await self._fetch_gdelt_emerging_articles(
                max_records, query_variant=query_variant
            )
            # Only trust these fields from GDELT; LLM fills rest (country/region/code/etc).
            articles: List[Dict[str, Any]] = []
            for a in articles_raw[:max_records]:
                if not isinstance(a, dict):
                    continue
                url = str(a.get("url", "")).strip()
                title = str(a.get("title", "")).strip()
                if not url.startswith(("http://", "https://")) or not title:
                    continue

                articles.append(
                    {
                        "url": url,
                        "title": title,
                        "seendate": str(a.get("seendate", "")).strip(),
                        "domain": str(a.get("domain", "")).strip(),
                        "language": str(a.get("language", "")).strip(),
                        "sourcecountry": str(a.get("sourcecountry", "")).strip(),
                        "socialimage": str(a.get("socialimage", "")).strip(),
                    }
                )

            if not articles:
                raise ValueError("Insufficient usable GDELT articles")

            system_prompt = AHIPromptTemplates.emerging_trend_risk_prompt()
            user_template = AHIPromptTemplates.emerging_trends_and_issues_user_prompt()

            raw = await self._llm_svc.invoke_chain(
                system_prompt=system_prompt,
                user_template=user_template,
                variables={
                    "current_date": now_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "articles_json": json.dumps(articles, ensure_ascii=False),
                },
                label="emerging-trends-and-issues",
            )

            analysis = json.loads(jrp.clean_json_response(raw))

            # Guardrail: ensure cards only reference provided URLs and titles.
            allowed_url_to_title = {a["url"]: a["title"] for a in articles if a.get("url")}
            cards = analysis.get("countries") or []
            if isinstance(cards, list):
                cleaned_cards: List[Dict[str, Any]] = []
                for c in cards:
                    if not isinstance(c, dict):
                        continue
                    u = str(c.get("sourceUrl", "")).strip()
                    t = str(c.get("title", "")).strip()
                    if not u or u not in allowed_url_to_title:
                        continue
                    if allowed_url_to_title[u] != t:
                        c["title"] = allowed_url_to_title[u]
                    cleaned_cards.append(c)
                analysis["countries"] = cleaned_cards

            if not analysis.get("updatedAt"):
                analysis["updatedAt"] = now_utc.strftime("%Y-%m-%dT%H:%M:%SZ")

            return {
                "success": True,
                "data": analysis,
            }

        except Exception as exc:
            logger.exception("emerging_trends_and_issues failed")

            return {
                "success": False,
                "error": str(exc),
            }


    async def pillar_live_signals(
        self,
        pillars: Dict[int, Dict[str, Any]],
    ) -> Dict[str, Any]:
        try:
            pillar_ids = sorted(pillars.keys())
            pillar_count = len(pillar_ids)
            id_range = (
                f"{pillar_ids[0]} through {pillar_ids[-1]}"
                if pillar_count > 1
                else str(pillar_ids[0]) if pillar_ids else "none"
            )
            system_prompt = AHIPPillarPrompts.pillar_live_signals_prompt(pillars)

            user_template = f"""
            Generate the LIVE African AHIP pillar signals feed (all {pillar_count} active pillars).

            Current UTC datetime (now):
            {{current_date}}

            Live coverage window start (48 hours before now):
            {{recency_cutoff}}

            Requirements:
            - Exactly {pillar_count} entries: pillarId {id_range}, each once.
            - Search each pillar domain before writing its card.
            - Use verified sourceUrl rules from the system prompt.
            """

            now_utc = datetime.now(timezone.utc)
            raw = await self._llm_svc.invoke_chain(
                system_prompt=system_prompt,
                user_template=user_template,
                variables={
                    "current_date": now_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "recency_cutoff": (now_utc - timedelta(hours=48)).strftime(
                        "%Y-%m-%dT%H:%M:%SZ"
                    ),
                },
                label="pillar-live-signals",
            )

            analysis = json.loads(jrp.clean_json_response(raw))

            return {
                "success": True,
                "data": analysis,
            }

        except Exception as exc:
            logger.exception("pillar_live_signals failed")

            return {
                "success": False,
                "error": str(exc),
            }


    # ------------------------------------------------------------------ #
    #  Helpers                                                            #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _build_context_block(chunks: List[Dict]) -> str:
        if not chunks:
            return ""
        lines = ["=== FROM UPLOADED COUNTRY DOCUMENTS ==="]
        for chunk in chunks:
            lines.append(f"[{chunk['section']}]\n{chunk['text']}\n")
        return "\n".join(lines)

    @staticmethod
    def _build_context_block(chunks: List[Dict]) -> str:
        if not chunks:
            return ""
        from app.services.common.url_verifier import is_valid_source_url

        catalog: List[str] = []
        lines = ["=== FROM UPLOADED COUNTRY DOCUMENTS ==="]
        source_idx = 0
        for chunk in chunks:
            url = str(chunk.get("source_url") or "").strip()
            prefix = ""
            if is_valid_source_url(url):
                source_idx += 1
                name = chunk.get("source_name") or chunk.get("file") or "RAG"
                title = chunk.get("file") or chunk.get("section") or ""
                catalog.append(f"source_{source_idx} | {name} | {title} | {url}")
                prefix = f"[source_{source_idx}] "
            lines.append(f"{prefix}[{chunk['section']}]\n{chunk['text']}\n")

        if catalog:
            lines.insert(
                0,
                "=== VERIFIED RAG SOURCES (copy source_id only; never invent URLs) ===\n"
                + "\n".join(catalog)
                + "\n=== END VERIFIED RAG SOURCES ===\n",
            )
        return "\n".join(lines)



rag_query_service = RAGQueryService()
