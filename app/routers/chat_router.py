"""
Score analysis Router - API endpoints with database exception logging
Fire-and-forget pattern for long-running analysis tasks
"""
import logging
from typing import Optional
from fastapi import APIRouter, HTTPException, Query
from app.view_models.ChatRequest import ChatCityExecutiveSlidesRequest, ChatCityExecutiveSlidesResponse, ChatCityRequest, ChatCrossComparisionRequest,ChatGlobalRequest, ChatRequest
from app.view_models.AnalysisRequest import ChatResponse
from app.view_models.CityExecutiveSlidesResult import CityExecutiveSlidesResult
from app.view_models.EmergingTrendsResult import ChatEmergingTrendsResponse
from app.view_models.PillarLiveSignalsResult import ChatPillarLiveSignalsResponse
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
        result = await chat_service.answer_city_question (
            city_id = request.cityID,
            questionText = request.questionText,
            historyText = request.historyText,
            faqid = request.faqid,
            pillar_id = request.pillarID 
        )

        return ChatResponse(
            success=True,
            message="Response fetched successfully",
            result=result
        )
    except Exception as e:
        logger.error(f"Error in chat API: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/city", response_model=ChatResponse)
async def ask(request: ChatCityRequest):
    """
    Chat endpoint:
    - Accepts user question in body
    - Runs RAG pipeline
    - Returns AI-generated answer
    """
    try:
        result = await chat_service.answer_city_question (
            city_id = request.cityID,
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
    
@router.post("/cross-comparision", response_model = ChatResponse)
async def ask(request: ChatCrossComparisionRequest):

    try:
        result = await chat_service.answer_crossComparision (
            questionText = request.questionText,
            cityIDs = request.cityIDs,
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
# ============================================================
# ROUTER
# ============================================================

@router.post(
    "/executive-slides",
    response_model=ChatCityExecutiveSlidesResponse
)
async def ask_city_executive_slides(
    request: ChatCityExecutiveSlidesRequest
):
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

        response = await chat_service.answer_city_executive_slides(
            city_id=request.cityId
        )

        return ChatCityExecutiveSlidesResponse(
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
    summary="Global emerging trends and issues feed",
)
async def get_emerging_trends_and_issues(
    cityCount: int = Query(
        default=8,
        ge=1,
        le=250,
        description="Number of GDELT articles to fetch (maxrecords); one card per article.",
    ),
    queryVariant: Optional[int] = Query(
        default=None,
        ge=0,
        description=(
            "GDELT keyword variant index (0–5). Omit to auto-rotate every 5 minutes. "
            "Each variant uses a different 2–3 keyword OR group, or all six terms."
        ),
    ),
):
    """
    Public homepage feed for emerging global risks and stability trends.

    Returns structured city cards suitable for a public-facing UI.
    """
    try:
        response = await chat_service.get_emerging_trends_and_issues(
            city_count=cityCount,
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
    summary="Live global VUI pillar signals (all 14 pillars)",
)
    

async def get_pillar_live_signals():
    """
    Public feed: one concise live signal per Verdian Urban Index pillar (IDs 1–14).
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