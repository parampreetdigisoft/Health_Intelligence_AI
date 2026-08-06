"""
Score analysis Router - API endpoints with database exception logging
Fire-and-forget pattern for long-running analysis tasks
"""
import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from app.view_models.ChatRequest import ChatCountryExecutiveSlidesRequest, ChatCountryExecutiveSlidesResponse, ChatCountryRequest, ChatCrossComparisionRequest, ChatGlobalRequest, ChatRequest
from app.view_models.AnalysisRequest import ChatResponse
from app.view_models.EmergingTrendsResult import ChatEmergingTrendsResponse
from app.view_models.PillarLiveSignalsResult import ChatPillarLiveSignalsResponse
from app.view_models.KpiSummaryRequest import KpiSummaryRequest, KpiSummaryResponse, KpiSummaryResult
logger = logging.getLogger(__name__)
from app.services.chat_service import chat_service


router = APIRouter(prefix="/api/chat", tags=["chat"])


@router.post("/ask", response_model=ChatResponse)
async def ask(request: ChatRequest):
    """
    Chat endpoint:
    - Accepts user question in body
    - Runs RAG pipeline
    - Returns AI-generated answer
    """
    try:
        result = await chat_service.answer_country_question (
            country_id = request.countryID,
            question = request.questionText,
            pillar_id = request.pillarID 
        )

        return ChatResponse (
            success=True,
            message="Response fetched successfully",
            result=result
        )
    except Exception as e:
        logger.error(f"Error in chat API: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/country", response_model=ChatResponse)
async def ask(request: ChatCountryRequest):
    """
    Chat endpoint:
    - Accepts user question in body
    - Runs RAG pipeline
    - Returns AI-generated answer
    """
    try:
        result = await chat_service.answer_country_question (
            country_id = request.countryID,
            questionText = request.questionText,
            historyText = request.historyText,
            faqid = request.faqid,
            pillar_id = request.pillarID 
        )

        return ChatResponse (
            success=True,
            message="Response fetched successfully",
            result=result
        )
    except Exception as e:
        logger.error(f"Error in chat API: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    
@router.post("/global", response_model = ChatResponse)
async def ask(request: ChatGlobalRequest):
    """
    Chat endpoint:
    - Accepts user question in body
    - Runs RAG pipeline
    - Returns AI-generated answer
    """
    try:
        result = await chat_service.answer_global_question (
            questionText = request.questionText,
            historyText = request.historyText, 
            faqid = request.faqid,
        )

        return ChatResponse(
            success=True,
            message="Response fetched successfully",
            result=result
        )
    except Exception as e:
        logger.error(f"Error in chat API: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    
@router.post("/cross-comparision", response_model = ChatResponse)
async def ask(request: ChatCrossComparisionRequest):

    try:
        result = await chat_service.answer_crossComparision (
            questionText = request.questionText,
            countryIDs = request.countryIDs,
            historyText = request.historyText
        )

        return ChatResponse(
            success=True,
            message="Response fetched successfully",
            result=result
        )
    except Exception as e:
        logger.error(f"Error in chat API: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    

@router.post(
    "/kpi-summary",
    response_model=KpiSummaryResponse,
    summary="Summarize KPI performance for end users",
)
async def summarize_kpi_performance(request: KpiSummaryRequest):
    """
    Sync KPI summarization endpoint.
    Accepts KPI purpose, scores (role-filtered by C#), and interpretation bands.
    Returns structured JSON summary scoped only to the provided KPI.
    """
    try:
        if not request.layerName or not request.layerCode:
            raise HTTPException(
                status_code=400,
                detail="layerName and layerCode are required",
            )

        response = await chat_service.summarize_kpi_performance(request)

        if not response.get("success"):
            return KpiSummaryResponse(
                success=False,
                message=response.get("message") or "Failed to generate KPI summary",
                result=None,
            )

        result_data = response.get("result") or {}
        return KpiSummaryResponse(
            success=True,
            message=response.get("message") or "KPI summary generated successfully",
            result=KpiSummaryResult(**result_data) if result_data else None,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in KPI summary API: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/executive-slides",response_model=ChatCountryExecutiveSlidesResponse)
async def ask_Country_executive_slides(request: ChatCountryExecutiveSlidesRequest):
    """
    Executive intelligence dashboard endpoint.

    Returns:
    - Daily performance
    - Weekly performance
    - Monthly performance
    - Combined risks
    - Early warnings
    """

    try:

        response = await chat_service.answer_Country_executive_slides(
            country_id=request.countryId
        )

        return ChatCountryExecutiveSlidesResponse(
            success=response["success"],
            message=response["message"],
            result=response["result"]
        )

    except Exception as e:

        logger.error(
            f"Error in executive slides API: {str(e)}",
            exc_info=True
        )

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


@router.get(
    "/emerging-trends-and-issues",
    response_model=ChatEmergingTrendsResponse,
    summary="Africa health emerging trends and risks feed",
)
async def get_emerging_trends_and_issues(
    countryCount: int = Query(
        default=8,
        ge=1,
        le=250,
        description="Number of GDELT articles to fetch (maxrecords); one card per article.",
    ),
    queryVariant: Optional[int] = Query(
        default=None,
        ge=0,
        description=(
            "GDELT health keyword variant index (0–7). Omit to auto-rotate every 5 minutes. "
            "Each variant uses different Africa-scoped health risk keywords."
        ),
    ),
):
    """
    Public homepage feed for emerging African health risks and trends.

    Fetches GDELT articles (last 24h) filtered for Africa and health-related topics,
    then returns structured country cards for the Health Intelligence UI.
    """
    try:
        response = await chat_service.get_emerging_trends_and_issues(
            country_count=countryCount,
            query_variant=queryVariant,
        )

        if not response.get("success"):
            raise HTTPException(
                status_code=502,
                detail=response.get("message", "Failed to generate emerging trends"),
            )

        return ChatEmergingTrendsResponse(
            success=True,
            message=response["message"],
            result=response["result"],
        )

    except HTTPException:
        raise

    except Exception as e:
        logger.error(
            f"Error in emerging trends API: {str(e)}",
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/pillar-live-signals",
    response_model=ChatPillarLiveSignalsResponse,
    summary="Live AHIP pillar signals (all active pillars)",
)
async def get_pillar_live_signals():
    """
    Public feed: one concise live signal per active Africa Health Intelligence pillar.
    """
    try:
        response = await chat_service.get_pillar_live_signals()

        if not response.get("success"):
            raise HTTPException(
                status_code=502,
                detail=response.get("message", "Failed to generate pillar live signals"),
            )

        return ChatPillarLiveSignalsResponse(
            success=True,
            message=response["message"],
            result=response["result"],
        )

    except HTTPException:
        raise

    except Exception as e:
        logger.error(
            f"Error in pillar live signals API: {str(e)}",
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail=str(e))
