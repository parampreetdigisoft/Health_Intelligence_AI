"""
Database Log Handler
---------------------
A standard logging.Handler that writes ERROR-level (and above) records
to the AppLogs table with an AI_ prefix on the level name.

This is wired into the root logger by setup_logging() in logging_config.py.
Do NOT call this directly in business code — just use:
    logger = logging.getLogger(__name__)
    logger.error("something broke")
"""

import logging
import traceback
from datetime import datetime
from typing import Optional

from app.services.core.connection import DBEngine, db_engine

_LEVEL_PREFIX = "AI_"


class DatabaseLogHandler(logging.Handler):
    """
    Logging handler that persists records to AppLogs.

    Only ERROR and above are written (enforced by setLevel in logging_config.py).
    Falls back to console print if the DB write itself fails — never raises.
    """

    def __init__(self, engine: DBEngine = None):
        super().__init__()
        self.engine = engine or db_engine

    async def emit(self, record: logging.LogRecord) -> None:
        try:
            exception_text: Optional[str] = None
            if record.exc_info:
                exception_text = "".join(traceback.format_exception(*record.exc_info))

            level      = _LEVEL_PREFIX + record.levelname   # e.g. AI_ERROR
            message    = self.format(record)
            created_at = datetime.fromtimestamp(record.created)

            query = """
                INSERT INTO AppLogs (Level, Message, Exception, CreatedAt)
                VALUES (?, ?, ?, ?)
            """
            await self.engine.execute_write_async(query, (level, message, exception_text, created_at))

        except Exception as e:
            # Never let a logging failure crash the app
            print(f"[DatabaseLogHandler] Failed to write log to DB: {e}")
            self.handleError(record)



            