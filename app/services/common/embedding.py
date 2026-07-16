"""
Offline-first embedding model loader for ChromaDB.

Loads sentence-transformers/all-MiniLM-L6-v2 from a local folder so the
service does not contact Hugging Face at runtime (required on locked-down servers).
"""
import logging
import os
from pathlib import Path

from chromadb.utils import embedding_functions

from app.config import settings

logger = logging.getLogger(__name__)

HF_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_FOLDER_NAME = "all-MiniLM-L6-v2"


def resolve_embedding_model_path() -> Path:
    path = Path(settings.EMBEDDING_MODEL_PATH)
    if not path.is_absolute():
        path = settings.BASE_DIR / path
    return path


def is_local_model_ready(path: Path) -> bool:
    return path.is_dir() and (
        (path / "config.json").is_file() or (path / "modules.json").is_file()
    )


def create_embedding_function():
    local_path = resolve_embedding_model_path()

    if is_local_model_ready(local_path):
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        logger.info("Loading embedding model from local path: %s", local_path)
        return embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=str(local_path)
        )

    allow_remote = (
        os.getenv("EMBEDDING_ALLOW_REMOTE_DOWNLOAD", "false").lower() == "true"
    )
    if allow_remote:
        logger.warning(
            "Local embedding model not found at %s; downloading from Hugging Face",
            local_path,
        )
        return embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=HF_MODEL_ID
        )

    raise FileNotFoundError(
        f"Embedding model not found at {local_path}. "
        "Download it once on a machine with internet:\n"
        "  python scripts/download_embedding_model.py\n"
        "Then copy the models/all-MiniLM-L6-v2 folder to the server, "
        "or set EMBEDDING_MODEL_PATH to its absolute path."
    )
