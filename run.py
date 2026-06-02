"""
FastAPI server launcher
Usage: python run.py
"""

import uvicorn
from app.config import settings

if __name__ == "__main__":
    print(
        f"""
        ╔══════════════════════════════════════════════╗
        ║   City Assessment AI Service                 ║
        ║   Starting server...                         ║
        ╚══════════════════════════════════════════════╝
        
        📍 API: http://{settings.API_HOST}:{settings.API_PORT}
        📖 Docs: http://{settings.API_HOST}:{settings.API_PORT}/docs
        🔍 Health: http://{settings.API_HOST}:{settings.API_PORT}/health
        
        Endpoints:
        • POST /api/chat/ask - Q&A chatbot
        • POST /api/scoring/evaluate - AI scoring
        • POST /api/summarizer/summarize - Text summary
        """
    )

    uvicorn.run(
        "app.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.API_RELOAD,
        log_level="info",
    )
