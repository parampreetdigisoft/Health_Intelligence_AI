# =========================================================================== #
#  chat_service.py  (refactored)                                         #
# =========================================================================== #

from datetime import datetime, timezone
import logging
import re
from typing import List, Dict, Any, Optional

from pydantic import ValidationError
from requests.compat import urlparse
from app.services.common.llm_base_service import LLMBaseService
from app.services.common.city_prompt import VerdianPromptTemplates
from app.services.common.pillar_prompts import PillarPrompts
from app.services.common.url_verifier import ensure_live_source_url
from app.services.common.url_verifier import ensure_live_source_url_pillars 
from app.services.core.repository import DatabaseRepository
from app.services.rag_query_service import rag_query_service
from app.view_models.PillarLiveSignalsResult import PillarLiveSignalsResult
from app.view_models.EmergingTrendsResult import EmergingTrendsResult

logger = logging.getLogger(__name__)

CHROMA_PATH = "./chroma_store"


class ChatService:
    """
    Hybrid RAG service: LLM-routed TOC selection + ChromaDB vector retrieval.

    LLM mechanics live in LLMBaseService (injected).
    Prompt text lives in VerdianPromptTemplates.
    """

    def __init__(self) -> None:
        self._db = DatabaseRepository()
        self._llm_svc = LLMBaseService(max_retries=3, retry_delay=1.0)

    async def initialize(self) -> None:
        """Initialise the shared LLM service."""
        await self._llm_svc.initialize()

    # ------------------------------------------------------------------ #
    #  Public Methods                                                    #
    # ------------------------------------------------------------------ #

    async def answer_city_question (
        self,
        city_id: int,
        questionText: str,
        historyText: Optional[str] = None,
        faqid : Optional[int] = None,
        pillar_id: Optional[int] = None,
    ) -> str:
        year = datetime.now().year      

        ai_city_context = await self._db.get_ai_city_context(city_id, year,pillar_id)

        if faqid is None :
            faqs = await self._db.get_FAQ_context()
            relevant_faq_ids = await rag_query_service.get_related_FAQ_IDs(questionText, faqs)

            if len(relevant_faq_ids)>0:
                relevant_faq_ids = relevant_faq_ids[: 3 if historyText == None else 2]
                ai_context = await self._db.GetLocalContextDataForLLM(relevant_faq_ids,city_id,pillar_id)
            else:
                ai_context = await rag_query_service.get_city_document_context(city_id,questionText, pillar_id)
        else:
             ai_context = await self._db.GetLocalContextDataForLLM([faqid],city_id,pillar_id)
            
        if len(ai_context) < 1:
            ai_context = "\n".join(f"{key}: {value}" for key, value in ai_city_context.items())
        pillar_name =ai_city_context["PillarName"]
        cityName =ai_city_context["CityName"]

        answer = await rag_query_service.send_city_question_to_llm(questionText,ai_context,cityName,pillar_name,historyText)

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

        cityName="global for all cities"
        pillar_name=""            

        answer = await rag_query_service.send_question_to_llm(questionText, ai_context, cityName, pillar_name, historyText)

        return answer
    # ============================================================
# CHAT SERVICE
# ============================================================

    async def answer_city_executive_slides( self, city_id: int) -> Dict[str, Any]:
        try:
            year = datetime.now().year

            ai_city = await self._db.get_ai_city_context(city_id, year)


            if not ai_city:
                return {
                    "success": False,
                    "message": "city context not found"
                }

            city_name = ai_city["CityName"]

            ai_city_context = "\n".join(
                f"{key}: {value}"
                for key, value in ai_city.items()
            )

            all_pillar_contexts = PillarPrompts.get_all_pillar_names()

            ai_result  = await rag_query_service.city_executive_slides(
                city_name=city_name,
                country=ai_city["Country"],
                ai_city_context=ai_city_context,
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
                "cityId": city_id,
                "cityName": data.get("cityName"),

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
                "answer_city_executive_slides_question failed"
            )

            return {
                "success": False,
                "error": str(exc)
            }
        
    async def answer_crossComparision(
        self,
        questionText: str,
        cityIDs: list[int],
        historyText: Optional[str] = None,
    ) -> str:

        year = datetime.now().year

        cities = []

        if len(cityIDs) > 0:
            query = f"""
                SELECT CityName, Country
                FROM Cities
                WHERE CityID IN ({",".join(map(str, cityIDs))})
            """

            cities = await self._db.engine.fetch_dicts_async(query)

        relevant_faq_ids = []

        if len(cityIDs) == 0:
            faqs = await self._db.get_FAQ_context(True)
            relevant_faq_ids = await rag_query_service.get_related_FAQ_IDs(
                questionText,
                faqs
            )
        else:
            relevant_faq_ids = cityIDs

        if len(relevant_faq_ids) > 0:
            ai_context = await self._db.GetCrossComparisionLocalContextDataForLLM(
                relevant_faq_ids
            )
        else:
            ai_context = await rag_query_service.get_global_document_context(
                questionText
            )

        cityName = ", ".join(
            [city["CityName"] for city in cities]
        )

        pillar_name = "Get pillars from provided context"

        answer = await rag_query_service.send_question_to_llm(
            questionText,
            ai_context,
            cityName,
            pillar_name,
            historyText
        )

        return answer
    
    async def get_emerging_trends_and_issues(
        self,
        city_count: int = 8,
        query_variant: Optional[int] = None,
    ) -> Dict[str, Any]:
        try:
            max_records = max(1, min(250, city_count))

            ai_result = await rag_query_service.emerging_trends_and_issues(
                city_count=max_records,
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

        cities_raw = data.get("cities") or []
        normalized_cities: List[Dict[str, Any]] = []

        for item in cities_raw:
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

            normalized_cities.append(
                {
                    "city": str(item.get("city", "")).strip(),
                    "cityCode": str(item.get("cityCode", "")).strip().upper()[:2],
                    "country": str(item.get("country", "")).strip(), 
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

        if not normalized_cities:
            raise ValueError("Insufficient city cards in LLM response")

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
            "cities": normalized_cities,
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
            ai_result = await rag_query_service.pillar_live_signals()

            if not ai_result.get("success"):
                return {
                    "success": False,
                    "message": "Failed to generate pillar live signals",
                }

            normalized = self._normalize_pillar_live_signals_payload(ai_result["data"])
            normalized = await self._verify_pillar_live_signals_urls(normalized)
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
    def _normalize_pillar_live_signals_payload(data: Dict[str, Any]) -> Dict[str, Any]:
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

            if pillar_id < 1 or pillar_id > 14:
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

        if len(by_id) < 14:
            raise ValueError(
                f"Expected 14 pillar cards, received {len(by_id)} valid entries"
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
                    "Global peace-enabler pillar watch from the last 48 hours.",
                )
            ).strip(),
            "pillars": pillars,
        }

    @staticmethod
    async def _verify_pillar_live_signals_urls(data: Dict[str, Any]) -> Dict[str, Any]:
        pillar_names = PillarPrompts.get_all_pillar_names()
        verified: List[Dict[str, Any]] = []

        for item in data.get("pillars") or []:
            if not isinstance(item, dict):
                continue

            pillar_id = int(item["pillarId"])
            pillar_name = pillar_names.get(pillar_id, f"Pillar {pillar_id}")
            title = str(item.get("title", "")).strip()
            url = str(item.get("sourceUrl", "")).strip()

            item["sourceUrl"] = await ensure_live_source_url_pillars(url, pillar_name, title)
            verified.append(item)

        if len(verified) < 14:
            raise ValueError("Insufficient pillar cards after URL verification")

        data["pillars"] = verified
        return data


chat_service = ChatService()
