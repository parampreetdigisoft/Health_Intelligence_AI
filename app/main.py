"""
Main FastAPI Application
"""

import logging
from app.services.core.logging_config import setup_logging

# Must be called before anything else creates a logger
setup_logging()

from fastapi import FastAPI, Request
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.utils import get_openapi
from fastapi.openapi.docs import get_swagger_ui_html

from app.config import settings
from app.services.core.repository import db_repository
from app.middleware.auth_middleware import APIKeyMiddleware
from app.routers.document_processor_router import router as document_processor_router
from app.routers.score_analysis_router import router as score_analysis_router
from app.routers.chat_router import router as chat_router

# Every module gets its own logger — no handler setup needed here
logger = logging.getLogger(__name__)

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

app = FastAPI(
    title="AHI AI Service",
    description="Analysis API. **Authentication Required**: Add your API key using the 'Authorize' button below.",
    version="1.0.0",
    docs_url=None,
    redoc_url=None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)
app.add_middleware(APIKeyMiddleware)


def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title="AHI AI Service",
        version="1.0.0",
        description="Analysis API with API Key Authentication",
        routes=app.routes,
    )
    openapi_schema["components"]["securitySchemes"] = {
        "APIKeyHeader": {
            "type": "apiKey",
            "in": "header",
            "name": "X-API-Key",
            "description": "Enter your API key",
        }
    }
    excluded_paths = ["/health", "/docs", "/redoc", "/openapi.json"]
    for path, path_item in openapi_schema["paths"].items():
        if path not in excluded_paths:
            for method in path_item.values():
                if isinstance(method, dict) and "security" not in method:
                    method["security"] = [{"APIKeyHeader": []}]

    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi


@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    return get_swagger_ui_html(
        openapi_url="/openapi.json",
        title="AHI AI Service - Swagger UI",
        swagger_js_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5.9.0/swagger-ui-bundle.js",
        swagger_css_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5.9.0/swagger-ui.css",
        swagger_ui_parameters={
            "persistAuthorization": True,
            "displayRequestDuration": True,
            "filter": True,
        },
    )


@app.get("/redoc", include_in_schema=False)
async def redoc_html():
    from fastapi.openapi.docs import get_redoc_html
    return get_redoc_html(openapi_url="/openapi.json", title="Assessment AI Service - ReDoc")


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Unhandled exceptions: logged to file (via NSSM) AND to DB (via handler)."""
    logger.error(
        f"Unhandled exception at {request.url.path}",
        exc_info=exc,
        extra={
            "method": request.method,
            "url": str(request.url),
            "client": request.client.host if request.client else "unknown",
        },
    )
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": str(exc),
            "path": request.url.path,
        },
    )


@app.on_event("startup")
async def startup_event():
    logger.info("Starting AI Microservice...")
    try:
        if not db_repository.engine.test_connection():
            logger.warning("Database connection failed — some features may not work")

        logger.info("All services initialized successfully!")
        logger.info(f"API docs at http://{settings.API_HOST}:{settings.API_PORT}/docs")
    except Exception as e:
        logger.error(f"Startup failed: {e}", exc_info=True)
        raise


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down AI Microservice...")


app.include_router(chat_router)
app.include_router(document_processor_router)
app.include_router(score_analysis_router)


@app.get("/", summary="API Root", tags=["General"])
async def root():
    return {
        "service": "AHI AI Service",
        "status": "running",
        "model_in_use": settings.LLM_PROVIDER,
        "routes": {
            "health_check": "/health",
            "documentation": f"http://{settings.API_HOST}:{settings.API_PORT}/docs",
        },
    }


@app.get("/health", summary="Health Check", tags=["Health"])
async def health_check():
    return {"status": "healthy", "database": settings.DB_NAME}