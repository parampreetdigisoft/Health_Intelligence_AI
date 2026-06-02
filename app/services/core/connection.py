"""
DB Connection & Core Execution Engine (Async Wrapper with ThreadPool)
----------------------------------------------------------------------
pyodbc + ThreadPool so async/await can be used safely.
SQLAlchemy used only for pd.read_sql() compatibility.
"""

import logging
import pyodbc
import pandas as pd
import asyncio
from contextlib import contextmanager
from typing import Any, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor

from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL

from app.config import settings

logger = logging.getLogger(__name__)

# Enable connection pooling
pyodbc.pooling = True

# ThreadPool (tune based on system)
executor = ThreadPoolExecutor(max_workers=20)


# ---------------------------------------------------------------------------
# Connection Factory  —  pyodbc (used by all write/SP/dict methods)
# ---------------------------------------------------------------------------

def build_connection_string(
    server: str = None,
    database: str = None,
    user: str = None,
    password: str = None,
    use_windows_auth: bool = None,
) -> str:

    server           = server           or settings.DB_SERVER
    database         = database         or settings.DB_NAME
    use_windows_auth = use_windows_auth if use_windows_auth is not None else settings.DB_USE_WINDOWS_AUTH

    base = (
        f"DRIVER={{ODBC Driver 17 for SQL Server}};"
        f"SERVER={server};"
        f"DATABASE={database};"
    )

    if use_windows_auth:
        return base + "Trusted_Connection=yes;"

    user     = user     or settings.DB_USER
    password = password or settings.DB_PASSWORD
    return base + f"UID={user};PWD={password};"


# ---------------------------------------------------------------------------
# Connection Factory  —  SQLAlchemy (used only by fetch_df)
# ---------------------------------------------------------------------------

def build_sqlalchemy_url(
    server: str = None,
    database: str = None,
    user: str = None,
    password: str = None,
    use_windows_auth: bool = None,
) -> URL:

    server           = server           or settings.DB_SERVER
    database         = database         or settings.DB_NAME
    use_windows_auth = use_windows_auth if use_windows_auth is not None else settings.DB_USE_WINDOWS_AUTH

    query_params = {"driver": "ODBC Driver 17 for SQL Server"}

    if use_windows_auth:
        query_params["Trusted_Connection"] = "yes"
        return URL.create(
            "mssql+pyodbc",
            host=server,
            database=database,
            query=query_params,
        )

    user     = user     or settings.DB_USER
    password = password or settings.DB_PASSWORD

    return URL.create(
        "mssql+pyodbc",
        username=user,
        password=password,
        host=server,
        database=database,
        query=query_params,
    )


# ---------------------------------------------------------------------------
# pyodbc Connection Context Manager
# ---------------------------------------------------------------------------

@contextmanager
def get_connection(connection_string: str, timeout: int = 30):
    conn = None
    try:
        conn = pyodbc.connect(connection_string, timeout=timeout)
        yield conn
    except pyodbc.Error as e:
        logger.error(f"DB connection error: {e}")
        raise
    finally:
        if conn:
            conn.close()


# ---------------------------------------------------------------------------
# Core Execution Engine
# ---------------------------------------------------------------------------

class DBEngine:

    def __init__(self, connection_string: str = None):
        self.connection_string = connection_string or build_connection_string()

        # SQLAlchemy engine — used exclusively by fetch_df for pd.read_sql()
        self._sa_engine = create_engine(
            build_sqlalchemy_url(),
            fast_executemany=True,
            pool_pre_ping=True,      # drops stale connections before use
            pool_size=10,
            max_overflow=10,
        )

    # ---------------- SYNC METHODS ---------------- #

    def execute_sp(self, sp_query: str, params: tuple) -> None:
        with get_connection(self.connection_string) as conn:
            try:
                cursor = conn.cursor()
                cursor.fast_executemany = True
                cursor.execute(sp_query, params)
                conn.commit()
            except Exception as e:
                conn.rollback()
                logger.exception(f"SP error — {sp_query} : {e}")
                raise

    def execute_write(
        self,
        query: str,
        params: tuple = (),
        *,
        fetch_one: bool = False,
        executemany: bool = False,
    ) -> Optional[Any]:

        with get_connection(self.connection_string) as conn:
            try:
                cursor = conn.cursor()

                if executemany:
                    cursor.fast_executemany = True
                    cursor.executemany(query, params)
                    result = None
                else:
                    cursor.execute(query, params)
                    result = cursor.fetchone() if fetch_one else None

                conn.commit()
                return result

            except Exception as e:
                conn.rollback()
                logger.exception(f"Write error — {query}: {e}")
                raise

    def fetch_dicts(
        self,
        query: str,
        params: tuple = None,
    ) -> List[Dict[str, Any]]:

        with get_connection(self.connection_string) as conn:
            cursor = conn.cursor()
            cursor.execute(query, params) if params else cursor.execute(query)

            columns = [col[0] for col in cursor.description]
            rows    = cursor.fetchall()

            return [dict(zip(columns, row)) for row in rows]

    def fetch_df(
        self,
        query: str,
        params: tuple = None,
    ) -> pd.DataFrame:
        """
        Uses SQLAlchemy connection to silence the pandas UserWarning
        about non-SQLAlchemy DBAPI2 connectors.
        Params are passed via SQLAlchemy's bindparams so injection is safe.
        """
        with self._sa_engine.connect() as conn:
            stmt = text(query)
            return (
                pd.read_sql(stmt, conn, params=params)
                if params
                else pd.read_sql(stmt, conn)
            )

    def test_connection(self) -> bool:
        try:
            with get_connection(self.connection_string) as conn:
                conn.cursor().execute("SELECT 1")
            return True
        except Exception:
            return False

    # ---------------- ASYNC METHODS ---------------- #

    async def execute_sp_async(self, sp_query: str, params: tuple) -> None:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            executor,
            self.execute_sp,
            sp_query,
            params,
        )

    async def execute_write_async(
        self,
        query: str,
        params: tuple = (),
        *,
        fetch_one: bool = False,
        executemany: bool = False,
    ) -> Optional[Any]:

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            executor,
            lambda: self.execute_write(
                query,
                params,
                fetch_one=fetch_one,
                executemany=executemany,
            ),
        )

    async def fetch_dicts_async(
        self,
        query: str,
        params: tuple = None,
    ) -> List[Dict[str, Any]]:

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            executor,
            self.fetch_dicts,
            query,
            params,
        )

    async def fetch_df_async(
        self,
        query: str,
        params: tuple = None,
    ) -> pd.DataFrame:

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            executor,
            self.fetch_df,
            query,
            params,
        )

    async def test_connection_async(self) -> bool:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            executor,
            self.test_connection,
        )

    # ---------------- UTIL ---------------- #

    @staticmethod
    def rows_to_tuples(rows: List[Dict], col_order: List[str]) -> List[tuple]:
        if not rows:
            return []

        df = pd.DataFrame(rows)

        for col in col_order:
            if col not in df.columns:
                df[col] = None

        return list(df[col_order].itertuples(index=False, name=None))


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

db_engine = DBEngine()