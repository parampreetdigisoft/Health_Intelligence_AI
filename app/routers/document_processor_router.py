"""
Score analysis Router - API endpoints with database exception logging
Fire-and-forget pattern for long-running analysis tasks
"""
import logging
import asyncio
from fastapi import APIRouter, HTTPException
from app.view_models.AnalysisRequest import AnalysisResponse
logger = logging.getLogger(__name__)
from app.services.read_process_document import read_process_document


router = APIRouter(prefix="/api/rag", tags=["Rag"])


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


@router.post("/process-document/{city_doc_id}", response_model = AnalysisResponse)
async def process_document(city_doc_id:int):
    """
    Read the document by it's id and save chunk in relation db and vector db
    Returns immediately while process document runs in background
    """
    try:
        # Start analysis in background
        asyncio.create_task(
            run_analysis_task(
                "process-document-by-city-documentid",
                read_process_document.process_document(city_doc_id)
            )
        )
        
        return AnalysisResponse(
            success=True,
            message="Document processing started successfully. Processing in background.",
        )
            
    except Exception as e:
        error_msg = f"Error process-document_{city_doc_id}: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/delete_document/{city_id}/{city_doc_id}", response_model=AnalysisResponse)
async def delete_document(city_id:int, city_doc_id:int):
    """
    Read the document by it's id and save chunk in relation db and vector db    
    Returns immediately while process document runs in background
    """
    try:

        await read_process_document.delete_document(city_id, city_doc_id)
        
        return AnalysisResponse(
            success=True,
            message="Deletion document started successfully. Processing in background.",
        )
            
    except Exception as e:
        error_msg = f"Error delete_document_{city_doc_id}: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
