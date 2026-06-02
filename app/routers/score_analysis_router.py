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

router = APIRouter(prefix="/api/cities-score-analysis", tags=["Score Analysis"])


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
async def analyze_all_cities_full():
    """
    Analyze table data and provide global summary for the assessment result for all cities
    Returns immediately while analysis runs in background
    """
    try:
        # Start analysis in background
        asyncio.create_task(
            run_analysis_task(
                "analyze_all_cities_full",
                score_analyzer_service.analyze_all_cities_questions()
            )
        )
        
        return AnalysisResponse(
            success=True,
            message="City analysis started successfully. Processing in background.",
        )
            
    except Exception as e:
        error_msg = f"Error starting city analysis: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/analyze/missing-pillar-questions",response_model=AnalysisResponse)
async def analyze_missing_pillar_questions(request: MissingPillarQuestionRequest):
    """
    Analyze only missing AI pillar question evaluations
    for a city and optional pillar.
    Runs in background.
    """

    try:
        city = await score_analyzer_service._get_city_data(
            request.cityID
        )

        if city.empty:
            raise HTTPException(
                status_code=404,
                detail=f"City not found: {request.cityID}"
            )
        asyncio.create_task(
            run_analysis_task(
                f"analyze_missing_pillar_questions_{request.cityID}_{request.pillarID}",
                score_analyzer_service.analyze_PillarQuestions(
                    city,
                    request.pillarID,
                    True
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

@router.post("/analyze/{city_id}/full", response_model=AnalysisResponse)
async def analyze_single_city_full(city_id: int):
    """
    Analyze table data and provide global summary for a single city
    Returns immediately while analysis runs in background
    """
    try:
        if not city_id:
            raise HTTPException(status_code=400, detail="City ID is required")
        
        # Start analysis in background
        asyncio.create_task(
            run_analysis_task(
                f"analyze_single_city_full_{city_id}",
                score_analyzer_service.analyze_all_cities_questions(city_id)
            )
        )
        
        return AnalysisResponse(
            success=True,
            message=f"City {city_id} analysis started successfully. Processing in background.",
        )
            
    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Error starting single city analysis (ID: {city_id}): {str(e)}"
        logger.error(error_msg, exc_info=True)
        
        raise HTTPException(status_code=500, detail=str(e))

    
@router.post("/analyze/{city_id}", response_model=AnalysisResponse)
async def analyze_single_City(city_id: int):
    """
    Analyze only the city summary (no pillars/questions)
    Returns immediately while analysis runs in background
    """
    try:
        if not city_id:
            raise HTTPException(status_code=400, detail="City ID is required")
        
        # Start analysis in background
        asyncio.create_task(
            run_analysis_task(
                f"analyze_single_city_{city_id}",
                score_analyzer_service.analyze_single_City(city_id)
            )
        )
        
        return AnalysisResponse(
            success=True,
            message=f"City {city_id} analysis started successfully. Processing in background.",
        )
            
    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Error starting single city analysis (ID: {city_id}): {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze/{city_id}/pillars", response_model=AnalysisResponse)
async def analyze_city_pillars(city_id: int):
    """
    Analyze pillars for a specific city
    Returns immediately while analysis runs in background
    """
    try:
        if not city_id:
            raise HTTPException(status_code=400, detail="City ID is required")
        
        # Start analysis in background
        asyncio.create_task(
            run_analysis_task(
                f"analyze_city_pillars_{city_id}",
                score_analyzer_service.analyze_city_pillars(city_id)
            )
        )
        
        return AnalysisResponse(
            success=True,
            message=f"City {city_id} pillar analysis started successfully. Processing in background.",
        )
            
    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Error starting pillar analysis (ID: {city_id}): {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

  
@router.post("/analyze/{city_id}/questions", response_model=AnalysisResponse)
async def analyze_questions_of_city(city_id: int):
    """
    Analyze all questions for all pillars of a city
    Returns immediately while analysis runs in background
    """
    try:
        if not city_id:
            raise HTTPException(status_code=400, detail="City ID is required")
        
        # Start analysis in background
        asyncio.create_task(
            run_analysis_task(
                f"analyze_questions_of_city_{city_id}",
                score_analyzer_service.analyze_questions_of_city_pillar(city_id)
            )
        )
        
        return AnalysisResponse(
            success=True,
            message=f"City {city_id} questions analysis started successfully. Processing in background.",
        )
            
    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Error starting questions analysis (ID: {city_id}): {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze/{city_id}/pillars/{pillar_id}/questions", response_model=AnalysisResponse)
async def analyze_questions_of_city_pillar(city_id: int, pillar_id: int):
    """
    Analyze all questions of a particular pillar for a city
    Returns immediately while analysis runs in background
    """
    try:
        if not city_id:
            raise HTTPException(status_code=400, detail="City ID is required")
        
        # Start analysis in background
        asyncio.create_task(
            run_analysis_task(
                f"analyze_questions_city_{city_id}_pillar_{pillar_id}",
                score_analyzer_service.analyze_questions_of_city_pillar(city_id, pillar_id)
            )
        )
        
        return AnalysisResponse(
            success=True,
            message=f"City {city_id} pillar {pillar_id} questions analysis started successfully. Processing in background.",
        )
            
    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Error starting pillar questions analysis (City: {city_id}, Pillar: {pillar_id}): {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    

    
@router.post("/analyze/{city_id}/single-pillar/{pillar_id}", response_model=AnalysisResponse)
async def analyze_single_pillar(city_id: int, pillar_id: int):
    """
    Analyze single pillar for a city
    Returns immediately while analysis runs in background
    """
    try:
        if not city_id and not pillar_id: 
            raise HTTPException(status_code=400, detail="provide required parameter")
        
        # Start analysis in background
        asyncio.create_task(
            run_analysis_task(
                f"analyze single city{city_id}_pillar_{pillar_id}",
                score_analyzer_service.analyze_Single_Pillar(city_id, pillar_id)
            )
        )
        
        return AnalysisResponse(
            success=True,
            message=f"City {city_id} pillar {pillar_id} analysis started successfully. Processing in background.",
        )
            
    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Error starting pillar analysis (City: {city_id}, Pillar: {pillar_id}): {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    
@router.post("/analyze/{city_id}/immediateSituation", response_model=AnalysisResponse)
async def analyze_immediateSituation(city_id: int):
    """
    Analyze single pillar for a city
    Returns immediately while analysis runs in background
    """
    try:
        if not city_id:
            raise HTTPException(status_code=400, detail="provide required parameter")

        # Start analysis in background
        asyncio.create_task(
            run_analysis_task(
                f"analyze single city{city_id}",
                score_analyzer_service.immediateSituation(city_id),
            )
        )

        return AnalysisResponse(
            success=True,
            message=f"City {city_id} analysis started successfully. Processing in background.",
        )

    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Error starting pillar analysis (City: {city_id}: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))   