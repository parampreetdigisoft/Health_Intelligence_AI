# app/middleware/auth_middleware.py
"""
API Key Authentication Middleware
"""

from fastapi import Request, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from app.config import settings
import logging

logger = logging.getLogger(__name__)


class APIKeyMiddleware(BaseHTTPMiddleware):
    """
    Middleware to validate API key in request headers
    """
    
    # Routes that don't require authentication
    EXCLUDED_PATHS = [
        "/docs",
        "/redoc",
        "/openapi.json",
        "/health",
    ]
    
    async def dispatch(self, request: Request, call_next):
        # Skip authentication for excluded paths
        if any(request.url.path.startswith(path) for path in self.EXCLUDED_PATHS):
            return await call_next(request)
        
        # Get API key from header
        api_key = request.headers.get("X-API-Key")  # or request.headers.get("Authorization")
        
        # # Remove "Bearer " prefix if present
        # if api_key and api_key.startswith("Bearer "):
        #     api_key = api_key[7:]
        
        # Validate API key
        if not api_key:
            logger.warning(f"Missing API key for {request.method} {request.url.path}")
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={
                    "error": "Unauthorized",
                    "message": "API key is missing. Please provide X-API-Key header."
                }
            )
        
        if api_key != settings.Application_Auth_API_KEY:
            logger.warning(f"Invalid API key attempt for {request.method} {request.url.path}")
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={
                    "error": "Unauthorized",
                    "message": "Invalid API key."
                }
            )
        
        # API key is valid, proceed with request
        response = await call_next(request)
        return response