# =========================================================================== #
#  chat_service.py  (refactored)                                              #
# =========================================================================== #

import json
import logging
import re
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse
from datetime import datetime, timezone
from pydantic import ValidationError
from app.services.core.repository import DatabaseRepository
from app.services.rag_query_service import rag_query_service
from app.services.common.llm_base_service import LLMBaseService
from app.services.common import json_response_parser as jrp
from app.services.common.pillar_prompts import AHIPPillarPrompts
from app.view_models.EmergingTrendsResult import EmergingTrendsResult
from app.view_models.PillarLiveSignalsResult import PillarLiveSignalsResult
from app.services.common.url_verifier import ensure_live_source_url
logger = logging.getLogger(__name__)
CHROMA_PATH = "./chroma_store"


class ChatService:


    def __init__(self) -> None:
        self._db = DatabaseRepository()
        self._llm_svc = LLMBaseService(max_retries=3, retry_delay=1.0)

    async def initialize(self) -> None:
        """Initialise the shared LLM service."""
        await self._llm_svc.initialize()

    # ------------------------------------------------------------------ #
    #  Public Methods                                                    #
    # ------------------------------------------------------------------ #

    async def answer_country_question (
        self,
        country_id: int,
        questionText: str,
        historyText: Optional[str] = None,
        faqid : Optional[int] = None,
        pillar_id: Optional[int] = None,
    ) -> str:
        year = datetime.now().year      

        ai_country_context = await self._db.get_ai_country_context(country_id, year,pillar_id)

        if faqid is None :
            faqs = await self._db.get_FAQ_context()
            relevant_faq_ids = await rag_query_service.get_related_FAQ_IDs(questionText, faqs)

            if len(relevant_faq_ids)>0:
                relevant_faq_ids = relevant_faq_ids[: 3 if historyText == None else 2]
                ai_context = await self._db.GetLocalContextDataForLLM(relevant_faq_ids,country_id,pillar_id)
            else:
                ai_context = await rag_query_service.get_country_document_context(country_id,questionText, pillar_id)
        else:
            ai_context = await self._db.GetLocalContextDataForLLM([faqid],country_id,pillar_id)
            
        if len(ai_context) < 1:
            ai_context = "\n".join(f"{key}: {value}" for key, value in ai_country_context.items())
        pillar_name =ai_country_context["PillarName"]
        countryName =ai_country_context["CountryName"]

        answer = await rag_query_service.send_question_to_llm(questionText,ai_context,countryName,pillar_name,historyText)

        return answer
    
    async def answer_global_question (
        self,
        questionText: str,
        historyText: Optional[str] = None,
        faqid: Optional[int] = None
    ) -> str:
        year = datetime.now().year    
        
        relevant_faq_ids =[]
        if faqid is None: 
            faqs = await self._db.get_FAQ_context(True)
            relevant_faq_ids = await rag_query_service.get_related_FAQ_IDs(questionText, faqs)
        else :
            relevant_faq_ids=[faqid]
            
        if len(relevant_faq_ids)>0:
            ai_context = await self._db.GetLocalContextDataForLLM(relevant_faq_ids)
        else:
            ai_context = await rag_query_service.get_global_document_context(questionText)

        countryName="global for all countries"
        pillar_name=""            

        answer = await rag_query_service.send_question_to_llm(questionText, ai_context, countryName, pillar_name, historyText)

        return answer

    
    async def answer_crossComparision(
        self,
        questionText: str,
        countryIDs: list[int],
        historyText: Optional[str] = None,
    ) -> str:

        year = datetime.now().year

        countries = []

        if len(countryIDs) > 0:
            query = f"""
                SELECT CountryName, Continent
                FROM Countries
                WHERE CountryID IN ({",".join(map(str, countryIDs))})
            """

            countries = await self._db.engine.fetch_dicts_async(query)

        relevant_faq_ids = []

        if len(countryIDs) == 0:
            faqs = await self._db.get_FAQ_context(True)
            relevant_faq_ids = await rag_query_service.get_related_FAQ_IDs(
                questionText,
                faqs
            )
        else:
            relevant_faq_ids = countryIDs

        if len(relevant_faq_ids) > 0:
            ai_context = await self._db.GetCrossComparisionLocalContextDataForLLM(
                relevant_faq_ids
            )
        else:
            ai_context = await rag_query_service.get_global_document_context(
                questionText
            )

        countryName = ", ".join(
            [country["CountryName"] for country in countries]
        )

        pillar_name = "Get pillars from provided context"

        answer = await rag_query_service.send_question_to_llm(
            questionText,
            ai_context,
            countryName,
            pillar_name,
            historyText
        )

        return answer
    

    async def answer_Country_executive_slides( self, country_id: int) -> Dict[str, Any]:
        try:
            year = datetime.now().year

            ai_country = await self._db.get_ai_country_context(country_id, year)

            if not ai_country:
                return {
                    "success": False,
                    "message": "country context not found"
                }

            country_name = ai_country["CountryName"]

            ai_country_context = "\n".join(
                f"{key}: {value}"
                for key, value in ai_country.items()
            )

            pillars = await self._db.get_active_pillars_map()
            all_pillar_contexts = AHIPPillarPrompts.get_all_pillar_names(pillars)

            ai_result  = await rag_query_service.country_executive_slides(
                country_name=country_name,
                ai_country_context=ai_country_context,
                allPillarContexts=all_pillar_contexts,
                year=year
            )

            if not ai_result.get("success"):
                return {
                    "success": False,
                    "message": "Failed to generate executive slides"
                }

            data = ai_result["data"]

            result = {
                "countryId": country_id,
                "countryName": data.get("countryName"),

                "recentPerformance": {
                    "trend": data["recentPerformance"]["trend"],
                    "summary": data["recentPerformance"]["summary"]
                },

                "combinedRisks": data["combinedRisks"]["risks"],

                "earlyWarnings": data["earlyWarnings"]["warnings"]
            }

            return {
                "success": True,
                "message": "Executive slides generated successfully",
                "result": result
            }

        except Exception as exc:

            logger.exception(
                "answer_country_executive_slides_question failed"
            )

            return {
                "success": False,
                "error": str(exc)
            }


    async def get_emerging_trends_and_issues(
        self,
        country_count: int = 8,
        query_variant: Optional[int] = None,
    ) -> Dict[str, Any]:
        try:
            max_records = max(1, min(250, country_count))

            ai_result = await rag_query_service.emerging_trends_and_issues(
                country_count=max_records,
                query_variant=query_variant,
            )

            if not ai_result.get("success"):
                return {
                    "success": False,
                    "message": "Failed to generate emerging trends and issues",
                }

            normalized = self._normalize_emerging_trends_payload(ai_result["data"])            
            validated = EmergingTrendsResult.model_validate(normalized)

            return {
                "success": True,
                "message": "Emerging trends and issues generated successfully",
                "result": validated.model_dump(),
            }

        except ValidationError as exc:
            logger.warning(
                "Emerging trends response failed validation: %s",
                exc,
            )
            return {
                "success": False,
                "message": "Emerging trends response did not meet quality checks",
            }

        except Exception as exc:
            logger.exception("get_emerging_trends_and_issues failed")

            return {
                "success": False,
                "message": str(exc),
            }


    @staticmethod
    def _normalize_emerging_trends_payload(data: Dict[str, Any]) -> Dict[str, Any]:
        category_map = {
            "governance": "Governance",
            "conflict": "Conflict",
            "economy": "Economy",
            "climate": "Climate",
            "security": "Security",
            "migration": "Migration",
            "society": "Society",
            "technology": "Technology",
            "health": "Health",
        }
        status_map = {
            "rising": "Rising",
            "active": "Active",
            "watch": "Watch",
            "stable": "Stable",
            "critical": "Critical",
        }
        icon_map = {
            "governance": "governance",
            "conflict": "conflict",
            "economy": "economy",
            "climate": "climate",
            "security": "security",
            "migration": "migration",
            "society": "society",
            "technology": "technology",
            "health": "health",
        }

        countries_raw = data.get("countries") or []
        normalized_countries: List[Dict[str, Any]] = []

        for item in countries_raw:
            if not isinstance(item, dict):
                continue

            category = str(item.get("category", "Governance")).strip()
            category_key = category.lower()
            category = category_map.get(category_key, category)
            if category not in category_map.values():
                category = "Governance"

            status = str(item.get("status", "Watch")).strip()
            status = status_map.get(status.lower(), status)
            if status not in status_map.values():
                status = "Watch"

            icon = str(item.get("icon", category_key or "governance")).strip().lower()
            icon = icon_map.get(icon, icon_map.get(category_key, "governance"))

            urgency = str(item.get("urgency", "medium")).strip().lower()
            card_type = str(item.get("type", "risk")).strip().lower()
            color = str(item.get("color", "yellow")).strip().lower()

            summary = " ".join(str(item.get("summary", "")).split())
            if len(summary) > 200:
                summary = summary[:180].rstrip() + "..."

            confidence = item.get("confidence", 70)
            try:
                confidence = int(confidence)
            except (TypeError, ValueError):
                confidence = 70
            confidence = max(0, min(100, confidence))

            source_url = ChatService._normalize_source_url(item)
            if not source_url:
                continue

            title = ChatService._strip_source_mentions(
                str(item.get("title", "")).strip()
            )
            summary = ChatService._strip_source_mentions(summary)

            normalized_countries.append(
                {
                    "country": str(item.get("country", "")).strip(),
                    "countryCode": str(item.get("countryCode", "")).strip().upper()[:2],
                    "region": str(item.get("region", "")).strip(),
                    "type": card_type if card_type in ("risk", "trend") else "risk",
                    "title": title,
                    "summary": summary,
                    "category": category,
                    "status": status,
                    "urgency": urgency if urgency in ("low", "medium", "high", "critical") else "medium",
                    "confidence": confidence,
                    "icon": icon,
                    "color": color if color in ("green", "yellow", "orange", "red", "blue") else "yellow",
                    "sourceUrl": source_url,
                }
            )

        if not normalized_countries:
            raise ValueError("Insufficient country cards in LLM response")

        updated_at = data.get("updatedAt")
        if not updated_at:
            updated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        return {
            "updatedAt": str(updated_at),
            "headline": str(data.get("headline", "Emerging Issues & Trends")).strip(),
            "subHeadline": str(
                data.get(
                    "subHeadline",
                    "Live global signals from the last 48 hours across governance, security, economy, and society.",
                )
            ).strip(),
            "countries": normalized_countries,
        }

    @staticmethod
    def _normalize_source_url(item: Dict[str, Any]) -> str:
        raw = item.get("sourceUrl") or item.get("source_url") or ""
        url = str(raw).strip()
        if not url:
            return ""

        if not url.startswith(("http://", "https://")):
            url = f"https://{url.lstrip('/')}"

        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            return ""

        return url

    @staticmethod
    def _strip_source_mentions(text: str) -> str:
        return re.sub(
            r"\s*(?:according to|reported by|sources? say|as reported|per reports?).*$",
            "",
            text.strip(),
            flags=re.IGNORECASE,
        ).strip()


    async def get_pillar_live_signals(self) -> Dict[str, Any]:
        try:
            pillars = await self._db.get_active_pillars_map()
            if not pillars:
                return {
                    "success": False,
                    "message": "No active pillars configured",
                }

            ai_result = await rag_query_service.pillar_live_signals(pillars)

            if not ai_result.get("success"):
                return {
                    "success": False,
                    "message": "Failed to generate pillar live signals",
                }

            normalized = self._normalize_pillar_live_signals_payload(
                ai_result["data"], pillars
            )
            normalized = await self._verify_pillar_live_signals_urls(normalized, pillars)
            validated = PillarLiveSignalsResult.model_validate(normalized)

            return {
                "success": True,
                "message": "Pillar live signals generated successfully",
                "result": validated.model_dump(),
            }

        except ValidationError as exc:
            logger.warning(
                "Pillar live signals response failed validation: %s",
                exc,
            )
            return {
                "success": False,
                "message": "Pillar live signals response did not meet quality checks",
            }

        except Exception as exc:
            logger.exception("get_pillar_live_signals failed")

            return {
                "success": False,
                "message": str(exc),
            }

    @staticmethod
    def _normalize_pillar_live_signals_payload(
        data: Dict[str, Any],
        pillars: Dict[int, Dict[str, Any]],
    ) -> Dict[str, Any]:
        expected_ids = set(pillars.keys())
        expected_count = len(expected_ids)
        status_map = {
            "rising": "Rising",
            "active": "Active",
            "watch": "Watch",
            "stable": "Stable",
            "critical": "Critical",
        }

        by_id: Dict[int, Dict[str, Any]] = {}

        for item in data.get("pillars") or []:
            if not isinstance(item, dict):
                continue

            try:
                pillar_id = int(item.get("pillarId", item.get("pillar_id", 0)))
            except (TypeError, ValueError):
                continue

            if pillar_id not in expected_ids:
                continue

            status = str(item.get("status", "Watch")).strip()
            status = status_map.get(status.lower(), status)
            if status not in status_map.values():
                status = "Watch"

            urgency = str(item.get("urgency", "medium")).strip().lower()
            color = str(item.get("color", "yellow")).strip().lower()
            card_type = str(item.get("type", "risk")).strip().lower()

            title = ChatService._strip_source_mentions(str(item.get("title", "")).strip())
            summary = ChatService._strip_source_mentions(
                " ".join(str(item.get("summary", "")).split())[:100]
            )
            source_url = ChatService._normalize_source_url(item)

            if not title or not summary or not source_url:
                continue

            by_id[pillar_id] = {
                "pillarId": pillar_id,
                "type": card_type if card_type in ("risk", "trend") else "risk",
                "title": title,
                "summary": summary,
                "status": status,
                "urgency": urgency if urgency in ("low", "medium", "high", "critical") else "medium",
                "color": color if color in ("green", "yellow", "orange", "red", "blue") else "yellow",
                "sourceUrl": source_url,
            }

        if len(by_id) < expected_count:
            raise ValueError(
                f"Expected {expected_count} pillar cards, received {len(by_id)} valid entries"
            )

        pillars = [by_id[pid] for pid in sorted(by_id.keys())]

        updated_at = data.get("updatedAt")
        if not updated_at:
            updated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        return {
            "updatedAt": str(updated_at),
            "headline": str(data.get("headline", "Live Pillar Signals")).strip(),
            "subHeadline": str(
                data.get(
                    "subHeadline",
                    "African health intelligence pillar watch from the last 48 hours.",
                )
            ).strip(),
            "pillars": pillars,
        }

    @staticmethod
    async def _verify_pillar_live_signals_urls(
        data: Dict[str, Any],
        pillars: Dict[int, Dict[str, Any]],
    ) -> Dict[str, Any]:
        pillar_names = AHIPPillarPrompts.get_all_pillar_names(pillars)
        expected_count = len(pillars)
        verified: List[Dict[str, Any]] = []

        for item in data.get("pillars") or []:
            if not isinstance(item, dict):
                continue

            pillar_id = int(item["pillarId"])
            pillar_name = pillar_names.get(pillar_id, f"Pillar {pillar_id}")
            title = str(item.get("title", "")).strip()
            url = str(item.get("sourceUrl", "")).strip()

            item["sourceUrl"] = await ensure_live_source_url(url, pillar_name, title)
            verified.append(item)

        if len(verified) < expected_count:
            raise ValueError("Insufficient pillar cards after URL verification")

        data["pillars"] = verified
        return data


chat_service = ChatService()
