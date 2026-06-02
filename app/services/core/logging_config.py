"""
Logging Configuration
----------------------
- INFO and above  → stdout  (NSSM pipes this to service.log)
- ERROR and above → stdout  (NSSM pipes stderr to error.log)
                  + DB handler (AppLogs table, level AI_ERROR)

Call setup_logging() once in main.py before the app is created.
After that, every module just does:
    logger = logging.getLogger(__name__)
"""

import logging
import sys
from app.services.core.logger import DatabaseLogHandler
from app.services.core.connection import db_engine


_CONSOLE_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT    = "%Y-%m-%d %H:%M:%S"

_logging_configured = False  # Guard against double-init


def setup_logging(db_level: int = logging.ERROR) -> None:
    """
    Configure the root logger.  Safe to call multiple times (no-op after first).

    Args:
        db_level: Minimum level written to the database (default: ERROR).
                  Pass logging.WARNING to also capture warnings in AppLogs.
    """
    global _logging_configured
    if _logging_configured:
        return

    # ------------------------------------------------------------------
    # Console handler — NSSM captures this into service.log / error.log
    # ------------------------------------------------------------------
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(_CONSOLE_FORMAT, _DATE_FORMAT))

    # ------------------------------------------------------------------
    # Database handler — errors only go to AppLogs table
    # ------------------------------------------------------------------
    db_handler = DatabaseLogHandler(engine=db_engine)
    db_handler.setLevel(db_level)
    db_handler.setFormatter(logging.Formatter(_CONSOLE_FORMAT, _DATE_FORMAT))

    # ------------------------------------------------------------------
    # Root logger — everything flows through here
    # ------------------------------------------------------------------
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(console_handler)
    root.addHandler(db_handler)

    # Silence noisy third-party loggers
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.error").setLevel(logging.WARNING)
    logging.getLogger("pyodbc").setLevel(logging.WARNING)

    _logging_configured = True