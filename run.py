"""
FastAPI server launcher
Usage: python run.py
"""

import uvicorn
from app.config import settings

if __name__ == "__main__":
    print("PEM AI Service Starting...")
    print(f"API: http://{settings.API_HOST}:{settings.API_PORT}")
    print(f"Docs: http://{settings.API_HOST}:{settings.API_PORT}/docs")
    print(f"Health: http://{settings.API_HOST}:{settings.API_PORT}/health")

    print("Endpoints:")
    print("POST /api/chat/ask - Q&A chatbot")
    print("POST /api/scoring/evaluate - AI scoring")
    print("POST /api/summarizer/summarize - Text summary")
 
    uvicorn.run(
        "app.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.API_RELOAD,
        log_level="info",
    )
