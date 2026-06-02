"""
Database Repository
--------------------
All domain/business queries live here.
Uses DBEngine for execution — never opens connections directly.
"""

import json
import logging
from typing import Any, Dict, List, Optional
import pandas as pd
from app.services.core.connection import DBEngine, db_engine

logger = logging.getLogger(__name__)

class DatabaseRepository:
    """
    Repository layer — owns every SQL query and stored-procedure call
    for the application domain.

    Injecting a custom `engine` makes testing / multi-tenant usage easy:
        repo = DatabaseRepository(engine=DBEngine(tenant_conn_string))
    """

    def __init__(self, engine: DBEngine = None):
        self.engine = engine or db_engine

    # ------------------------------------------------------------------
    # Views / generic reads
    # ------------------------------------------------------------------

    async def get_view_data(
        self,
        view_name: str,
        where: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """SELECT (optionally filtered) rows from a database view."""
        query = f"SELECT * FROM {view_name}"
        if limit:
            query = query.replace("SELECT", f"SELECT TOP {limit}", 1)
        if where:
            query += f" WHERE {where}"

        return await self.engine.fetch_df_async(query)

    # ------------------------------------------------------------------
    # Question evaluations
    # ------------------------------------------------------------------

    async def bulk_upsert_question_evaluations(self, rows: List[Dict]) -> None:
        if not rows:
            return

        col_order = [
                "CityID",
                "PillarID",
                "QuestionID",
                "Year",
                "AIScore",
                "AIProgress",
                "EvaluatorProgress",
                "Discrepancy",
                "ConfidenceLevel",
                "DataSourcesUsed",
                "EvidenceSummary",
                "RedFlags",
                "GeographicEquityNote",
                "SourceType",
                "SourceName",
                "SourceURL",
                "SourceDataYear",
                "SourceDataExtract",
                "SourceTrustLevel"
            ]


        records =  self.engine.rows_to_tuples(rows, col_order)
        await self.engine.execute_sp_async(
            "{CALL usp_AiBulkUpsertPillarQuestionEvaluations (?)}",
            (records,),
        )

    # ------------------------------------------------------------------
    # Pillar evaluations
    # ------------------------------------------------------------------

    async def bulk_upsert_pillar_evaluations(
        self,
        rows: List[Dict],
        sub_rows: List[Dict],
    ) -> None:
        if not rows:
            return
        
        score_df = pd.DataFrame(rows)[[
                "CityID",
                "PillarID",
                "Year",
                "AIScore",
                "AIProgress",
                "EvaluatorProgress",
                "Discrepancy",
                "ConfidenceLevel",
                "EvidenceSummary",
                "RedFlags",
                "GeographicEquityNote",
                "InstitutionalAssessment",
                "DataGapAnalysis",
                "AnalystDataGapAnalysis"
            ]]

        score_records = list(score_df.itertuples(index=False, name=None))
            
        source_df = pd.DataFrame(sub_rows)[[
                "CityID",
                "DataYear",
                "PillarID",
                "SourceType",
                "SourceName",
                "SourceURL",
                "DataExtract",
                "TrustLevel"
            ]]

        source_records = list(source_df.itertuples(index=False, name=None))
        
        await self.engine.execute_sp_async(
            "{CALL usp_AiBulkUpsertCityPillarEvaluations (?, ?)}",
            (score_records, source_records),
        )

    # ------------------------------------------------------------------
    # City evaluations
    # ------------------------------------------------------------------

    async def bulk_upsert_city_evaluations(self, rows: List[Dict]) -> None:
        if not rows:
            return

        col_order = [
                "CityID",
                "Year",
                "AIScore",
                "AIProgress",
                "EvaluatorProgress",
                "Discrepancy",
                "ConfidenceLevel",
                "EvidenceSummary",
                "CrossPillarPatterns",
                "InstitutionalCapacity",
                "EquityAssessment",
                "SustainabilityOutlook",
                "StrategicRecommendations",
                "DataTransparencyNote"
            ]

        records = self.engine.rows_to_tuples(rows, col_order)
        await self.engine.execute_sp_async(
            "EXEC usp_AiBulkUpsertCityEvaluations @CityEvaluations = ?",
            (records,),
        )

    # ------------------------------------------------------------------
    # Document TOC
    # ------------------------------------------------------------------

    async def save_toc_section(
        self,
        section: Dict,
        city_doc_id: int,
        city_id: int,
        pillar_id: Optional[int],
    ) -> Optional[int]:
        if not section:
            raise ValueError("section data is required")

        query = """
            MERGE DocumentTOC AS target
            USING (
                SELECT ? AS CityDocumentID,
                    ? AS CityID,
                    ? AS PillarID,
                    ? AS SectionPath,
                    ? AS SectionTitle,
                    ? AS SectionLevel,
                    ? AS PageStart,
                    ? AS PageEnd
            ) AS source
            ON target.CityDocumentID = source.CityDocumentID
            AND target.CityID = source.CityID
            AND (
                    (target.PillarID IS NULL AND source.PillarID IS NULL)
                    OR target.PillarID = source.PillarID
            )

            WHEN MATCHED THEN
                UPDATE SET
                    SectionTitle = source.SectionTitle,
                    SectionLevel = source.SectionLevel,
                    PageStart = source.PageStart,
                    PageEnd = source.PageEnd,
                    SectionPath=source.SectionPath

            WHEN NOT MATCHED THEN
                INSERT (CityDocumentID, CityID, PillarID, SectionPath,
                        SectionTitle, SectionLevel, PageStart, PageEnd)
                VALUES (source.CityDocumentID, source.CityID, source.PillarID,
                        source.SectionPath, source.SectionTitle,
                        source.SectionLevel, source.PageStart, source.PageEnd)

            OUTPUT inserted.TOCID;
            """

        params = (
            city_doc_id,
            city_id,
            pillar_id,
            section.get("path"),
            section.get("title"),
            section.get("level"),
            section.get("page_start"),
            section.get("page_end"),
        )

        result = await self.engine.execute_write_async(query, params, fetch_one=True)
        return result[0] if result else None

    # ------------------------------------------------------------------
    # Document chunks
    # ------------------------------------------------------------------

    async def save_document_chunks(
        self,
        chunks: List[Dict],
        city_doc_id: int,
        city_id: int,
        pillar_id: Optional[int],
    ) -> None:
        if not chunks:
            return

        query = """
            INSERT INTO DocumentChunks
                (ChunkID, CityDocumentID, TOCID, CityID, PillarID,
                 ChunkIndex, ChunkText)
            VALUES (?,?,?,?,?,?,?)
        """
        params = [
            (
                c.get("chunk_id"),
                city_doc_id,
                c.get("toc_id"),
                city_id,
                pillar_id,
                c.get("chunk_index"),
                c.get("chunk_text"),
            )
            for c in chunks
        ]

        await self.engine.execute_write_async(query, params, executemany=True)

    def test_connection(self) -> bool:
       return self.engine.test_connection()

    async def get_ai_city_context(
    self,
    city_id: int,
    year: int,
    pillar_id: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:

        query = """
                 SELECT 
            c.CityName,
            c.Country,
            a.AIScore,
            a.AIProgress,
            a.EvaluatorProgress,
            a.Discrepancy,
            a.ConfidenceLevel,
            a.EvidenceSummary,
            a.CrossPillarPatterns,
            a.InstitutionalCapacity,
            a.EquityAssessment,
            a.SustainabilityOutlook,
            a.StrategicRecommendations,
            a.DataTransparencyNote,
            a.ImmediateSituationSummary,
            a.KeyDevelopments,
            a.CriticalRisks,
            a.Gaps,
            a.UpdatedAt,
            a.IsVerified,
            a.VerifiedBy,
            p.PillarName
        FROM Cities c
        LEFT JOIN Pillars p
            ON p.PillarID = ?
        LEFT JOIN AICityScores a
            ON a.CityID = c.CityID
            AND a.Year = ?
        WHERE c.CityID = ?
        AND c.IsDeleted = 0
        """

        params = (pillar_id, year, city_id)

        result = await self.engine.fetch_dicts_async(query, params)

        return result[0] if result else None
        
    async def save_immediate_situation_summary(
        self,
        city_id: int,
        year: int,
        record: dict
    ) -> None:

        if not record:
            return

        query = """
            UPDATE AICityScores
            SET 
                ImmediateSituationSummary = ?,
                KeyDevelopments = ?,
                CriticalRisks = ?,
                Gaps = ?,
                EvidenceSummary = CASE 
                    WHEN ? IS NOT NULL AND LTRIM(RTRIM(CAST(? AS NVARCHAR(MAX)))) <> '' 
                    THEN ? 
                    ELSE EvidenceSummary 
                END
            WHERE CityID = ?
            AND Year = ?
        """

        exec_summary = record.get("executive_summary")

        params = (
            record.get("immediateSituationSummary"),
            record.get("key_developments"),
            record.get("critical_risks"),
            record.get("gaps"),
            exec_summary,   # check NULL
            exec_summary,   # check empty
            exec_summary,   # value to update
            city_id,
            year
        )

        await self.engine.execute_write_async(query, params)


    async def get_FAQ_context(self, isglobal: bool = False) -> List[Dict]:

        where = "WHERE Related LIKE 'global'" if isglobal else "WHERE Related LIKE 'city'"

        query = f"""
            SELECT FAQID, Related, Category, QuestionText,ProcName
            FROM AIAssistantFAQ
            {where}
        """

        return await self.engine.fetch_dicts_async(query)

    async def usp_GetCityDataForLLM(self, city_id: int, FAQIDs: List[str], pillarId: Optional[int] = None) -> List[Dict]:

        query = """
            EXEC dbo.usp_GetCityDataForLLM ?, ?, ?
        """

        params = (
           city_id, json.dumps(FAQIDs), pillarId
        )
        response = await self.engine.fetch_dicts_async(query, params)

        return response
    
    async def usp_GetGlobalDataForLLM(self, FAQIDs: List[str]) -> List[Dict]:
        query = """
            EXEC dbo.usp_GetGlobalDataForLLM ?
        """
        params = ( json.dumps(FAQIDs))
        response = await self.engine.fetch_dicts_async(query, params)

        return response

    async def GetLocalContextDataForLLM(self, FAQIDs: List[str],city_id: Optional[int] = None,  pillarId: Optional[int] = None) -> List[Dict]:

        query = """
            EXEC dbo.usp_GetLocalContextDataForLLM ?, ?, ?
        """

        params = (
            json.dumps(FAQIDs), city_id, pillarId
        )
        response = await self.engine.fetch_dicts_async(query, params)

        return response
    
    async def AiRecalculateCityScore(self, cityID: int) -> None:

        await self.engine.execute_sp_async(
            "EXEC sp_AiRecalculateCityScore @CityID = ?",
            (cityID,),
        )

    async def AiInsertAnalyticalLayerResults(self, cityID: int) -> None:

        await self.engine.execute_sp_async(
            "EXEC sp_AiInsertAnalyticalLayerResults @CityID = ?",
            (cityID,),
        )

    async def GetCrossComparisionLocalContextDataForLLM(self, city_ids: List[str]) -> List[Dict]:

        query = """
            EXEC dbo.usp_CityCrossComparision_faq ?
        """

        params = (
            json.dumps(city_ids)
        )
        response = await self.engine.fetch_dicts_async(query, params)

        return response    
# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

db_repository = DatabaseRepository()