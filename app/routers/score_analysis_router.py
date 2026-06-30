"""
Score analysis Router - API endpoints with database exception logging
Fire-and-forget pattern for long-running analysis tasks
"""

import logging
import asyncio
from typing import Optional
from fastapi import APIRouter, HTTPException
from app.view_models.MissingPillarQuestionRequest import MissingPillarQuestionRequest
from app.view_models.AnalysisRequest import AnalysisResponse
from app.services.score_analyzer_service import score_analyzer_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/countries-score-analysis", tags=["Score Analysis"])


# Background task wrapper with error handling
async def run_analysis_task(task_name: str, coro):
    """
    Wrapper to run analysis tasks in background with proper error handling
    """
    try:
        await coro

    except Exception as e:
        error_msg = f"Background task '{task_name}' failed: {str(e)}"
        logger.error(error_msg, exc_info=True)


@router.post("/analyze/full", response_model=AnalysisResponse)
async def analyze_all_countries_full():
    """
    Analyze table data and provide global summary for the assessment result for all countries
    Returns immediately while analysis runs in background
    """
    try:
        # Start analysis in background
        asyncio.create_task(
            run_analysis_task(
                "analyze_all_countries_full",
                score_analyzer_service.analyze_all_countries(),
            )
        )

        return AnalysisResponse(
            success=True,
            message="Country analysis started successfully. Processing in background.",
        )

    except Exception as e:
        error_msg = f"Error starting country analysis: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze/missing-pillar-questions",response_model=AnalysisResponse)
async def analyze_missing_pillar_questions(request: MissingPillarQuestionRequest):
    """
    Analyze only missing AI pillar question evaluations
    for a country and optional pillar.
    Runs in background.
    """

    try:

        asyncio.create_task(
            run_analysis_task(
                f"analyze_missing_pillar_questions_{request.countryID}",
                score_analyzer_service.import_missing_country_questions(
                    country_id=request.countryID,
                    pillar_id=request.pillarID
                )
            )
        )

        return AnalysisResponse(
            success=True,
            message=(
                "Missing pillar question analysis started "
                "successfully in background."
            ),
        )

    except HTTPException:
        raise

    except Exception as e:
        error_msg = (
            f"Error starting missing pillar question analysis: {str(e)}"
        )

        logger.error(error_msg, exc_info=True)

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
    
@router.post("/analyze/{country_id}/full", response_model=AnalysisResponse)
async def analyze_single_country_full(country_id: int):
    """
    Analyze table data and provide global summary for a single Country
    Returns immediately while analysis runs in background
    """
    try:
        if not country_id:
            raise HTTPException(status_code=400, detail="Country ID is required")

        # Start analysis in background
        asyncio.create_task(
            run_analysis_task(
                f"analyze_single_country_full_{country_id}",
                score_analyzer_service.analyze_all_countries(country_id),
            )
        )

        return AnalysisResponse(
            success=True,
            message=f"Country {country_id} analysis started successfully. Processing in background.",
        )

    except HTTPException:
        raise
    except Exception as e:
        error_msg = (
            f"Error starting single country analysis (ID: {country_id}): {str(e)}"
        )
        logger.error(error_msg, exc_info=True)

        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze/{country_id}", response_model=AnalysisResponse)
async def analyze_single_country(country_id: int):
    """
    Analyze only the country summary (no pillars/questions)
    Returns immediately while analysis runs in background
    """
    try:
        if not country_id:
            raise HTTPException(status_code=400, detail="Country ID is required")

        # Start analysis in background
        asyncio.create_task(
            run_analysis_task(
                f"analyze_single_country_{country_id}",
                score_analyzer_service.analyze_single_country(country_id),
            )
        )

        return AnalysisResponse(
            success=True,
            message=f"Country {country_id} analysis started successfully. Processing in background.",
        )

    except HTTPException:
        raise
    except Exception as e:
        error_msg = (
            f"Error starting single country analysis (ID: {country_id}): {str(e)}"
        )
        logger.error(error_msg, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze/{country_id}/pillars", response_model=AnalysisResponse)
async def analyze_country_pillars(country_id: int):
    """
    Analyze pillars for a specific country
    Returns immediately while analysis runs in background
    """
    try:
        if not country_id:
            raise HTTPException(status_code=400, detail="Country ID is required")

        # Start analysis in background
        asyncio.create_task(
            run_analysis_task(
                f"analyze_country_pillars_{country_id}",
                score_analyzer_service.analyze_country_pillars(country_id),
            )
        )

        return AnalysisResponse(
            success=True,
            message=f"Country {country_id} pillar analysis started successfully. Processing in background.",
        )

    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Error starting pillar analysis (ID: {country_id}): {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze/{country_id}/questions", response_model=AnalysisResponse)
async def analyze_questions_of_country(country_id: int):
    """
    Analyze all questions for all pillars of a country
    Returns immediately while analysis runs in background
    """
    try:
        if not country_id:
            raise HTTPException(status_code=400, detail="Country ID is required")

        # Start analysis in background
        asyncio.create_task(
            run_analysis_task(
                f"analyze_questions_of_country_{country_id}",
                score_analyzer_service.analyze_country_questions(country_id),
            )
        )

        return AnalysisResponse(
            success=True,
            message=f"Country {country_id} questions analysis started successfully. Processing in background.",
        )

    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Error starting questions analysis (ID: {country_id}): {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/analyze/{country_id}/pillars/{pillar_id}/questions",
    response_model=AnalysisResponse,
)
async def analyze_questions_of_country_pillar(country_id: int, pillar_id: int):
    """
    Analyze all questions of a particular pillar for a country
    Returns immediately while analysis runs in background
    """
    try:
        if not country_id:
            raise HTTPException(status_code=400, detail="Country ID is required")

        # Start analysis in background
        asyncio.create_task(
            run_analysis_task(
                f"analyze_questions_country_{country_id}_pillar_{pillar_id}",
                score_analyzer_service.analyze_country_questions(
                    country_id, pillar_id
                ),
            )
        )

        return AnalysisResponse(
            success=True,
            message=f"Country {country_id} pillar {pillar_id} questions analysis started successfully. Processing in background.",
        )

    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Error starting pillar questions analysis (Country: {country_id}, Pillar: {pillar_id}): {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze/{country_id}/single-pillar/{pillar_id}", response_model=AnalysisResponse)
async def analyze_single_pillar(country_id: int, pillar_id: int):
    """
    Analyze single pillar for a country
    Returns immediately while analysis runs in background
    """
    try:
        if not country_id and not pillar_id:
            raise HTTPException(status_code=400, detail="provide required parameter")

        # Start analysis in background
        asyncio.create_task(
            run_analysis_task(
                f"analyze single country{country_id}_pillar_{pillar_id}",
                score_analyzer_service.analyze_country_pillars(country_id, pillar_id),
            )
        )

        return AnalysisResponse(
            success=True,
            message=f"Country {country_id} pillar {pillar_id} analysis started successfully. Processing in background.",
        )

    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Error starting pillar analysis (Country: {country_id}, Pillar: {pillar_id}): {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/analyze/{country_id}/immediateSituation", response_model=AnalysisResponse)
async def analyze_immediateSituation(country_id: int):
    """
    Analyze single pillar for a country
    Returns immediately while analysis runs in background
    """
    try:
        if not country_id:
            raise HTTPException(status_code=400, detail="provide required parameter")

        # Start analysis in background
        asyncio.create_task(
            run_analysis_task(
                f"analyze single country{country_id}",
                score_analyzer_service.immediateSituation(country_id),
            )
        )

        return AnalysisResponse(
            success=True,
            message=f"Country {country_id} analysis started successfully. Processing in background.",
        )

    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Error starting pillar analysis (Country: {country_id}: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
