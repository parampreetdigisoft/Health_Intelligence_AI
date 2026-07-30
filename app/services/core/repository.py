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

_PILLAR_QUESTION_COUNTRY_EVAL_TVP_COLUMNS = [
    ("CountryID", "int"),
    ("PillarID", "int"),
    ("QuestionID", "int"),
    ("Year", "int"),
    ("AIScore", "decimal(5,2)"),
    ("AIProgress", "decimal(5,2)"),
    ("EvaluatorScore", "decimal(5,2)"),
    ("Discrepancy", "decimal(5,2)"),
    ("ConfidenceLevel", "varchar(200)"),
    ("EvidenceSummary", "nvarchar(max)"),
    ("StructuralEvidence", "nvarchar(max)"),
    ("OperationalEvidence", "nvarchar(max)"),
    ("OutcomeEvidence", "nvarchar(max)"),
    ("PerceptionEvidence", "nvarchar(max)"),
    ("TemporalScope", "nvarchar(max)"),
    ("DistortionScreening", "nvarchar(max)"),
    ("RelationalDependencies", "nvarchar(max)"),
    ("StressPoliticalShock", "nvarchar(max)"),
    ("StressEconomicShock", "nvarchar(max)"),
    ("StressNarrativeShock", "nvarchar(max)"),
    ("StressOverallResilienceShock", "nvarchar(max)"),
    ("InequalityAdjustment", "nvarchar(max)"),
    ("OpacityRisk", "nvarchar(max)"),
    ("RedFlag", "nvarchar(max)"),
    ("SourceName", "varchar(max)"),
    ("SourceType", "varchar(200)"),
    ("SourceURL", "varchar(max)"),
    ("SourceDataYear", "int"),
    ("SourceHierarchyLevel", "int"),
    ("SourceDataExtract", "nvarchar(max)"),
    ("SourcesConsulted", "int"),
]

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
    # score and Kpi recalculation
    # ------------------------------------------------------------------

    async def AiRecalculateCountryScore(self, countryID: int) -> None:

        await self.engine.execute_sp_async(
            "EXEC sp_AiRecalculateCountryScore @CountryID = ?",
            (countryID,),
        )

    async def AiInsertAnalyticalLayerResults(self, countryID: int) -> None:

        await self.engine.execute_sp_async(
            "EXEC sp_AiInsertAnalyticalLayerResults @CountryID = ?",
            (countryID,),
        )
    # ------------------------------------------------------------------
    # Question evaluations
    # ------------------------------------------------------------------


    async def bulk_upsert_question_evaluations(self, rows: List[Dict], countryID: int) -> None:
        if not rows:
            return

        col_order = [name for name, _ in _PILLAR_QUESTION_COUNTRY_EVAL_TVP_COLUMNS]

        records = []
        for row in rows:
            record = {name: row.get(name) for name in col_order}
            records.append(record)

        await self.engine.execute_sp_tvp_via_openjson_async(
            sp_name="usp_AiBulkUpsertPillarQuestionCountryEvaluations",
            tvp_type="dbo.TVP_PillarQuestionCountryEvaluationType",
            tvp_param_name="Evaluations",
            columns=_PILLAR_QUESTION_COUNTRY_EVAL_TVP_COLUMNS,
            rows=records,
        )

        await self.AiRecalculateCountryScore(countryID)

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

        await self.engine.execute_sp_async(
            "{CALL usp_AiBulkUpsertCountryPillarEvaluations (?, ?)}",
            (json.dumps(rows), json.dumps(sub_rows or [])),
        )

    # ------------------------------------------------------------------
    # Country evaluations
    # ------------------------------------------------------------------

    async def bulk_upsert_country_evaluations(self, rows: List[Dict]) -> None:
        if not rows:
            return

        col_order = [
            "CountryID", "Year", "AIScore", "AIProgress",
            "EvaluatorScore", "Discrepancy", "ConfidenceLevel",
            "EvidenceSummary", "StructuralEvidence",
            "OperationalEvidence", "OutcomeEvidence", "PerceptionEvidence",
            "TemporalScope", "DistortionScreening",
            "PoliticalShock", "EconomicShock", "NarrativeShock",
            "OverallStressResilience", "StressScoreAdjustment",
            "InequalityAdjustment", "OpacityRisk", "NonCompensationNote",
            "CrossPillarPatterns", "RelationalIntegrity",
            "InstitutionalCapacity", "EquityAssessment",
            "ConflictRiskOutlook", "StrategicRecommendation",
            "DataTransparencyNote", "PrimarySource",
        ]

        records = self.engine.rows_to_tuples(rows, col_order)
        await self.engine.execute_sp_async(
            "EXEC usp_AiBulkUpsertCountryEvaluations @CountryEvaluations = ?",
            (records,),
        )

    # ------------------------------------------------------------------
    # Document TOC
    # ------------------------------------------------------------------

    async def save_toc_section(
        self,
        section: Dict,
        country_doc_id: int,
        country_id: Optional[int],
        pillar_id: Optional[int],
    ) -> Optional[int]:
        if not section:
            raise ValueError("section data is required")

        query = """
            MERGE DocumentTOC AS target
            USING (
                SELECT ? AS CountryDocumentID,
                    ? AS CountryID,
                    ? AS PillarID,
                    ? AS SectionPath,
                    ? AS SectionTitle,
                    ? AS SectionLevel,
                    ? AS PageStart,
                    ? AS PageEnd
            ) AS source
            ON target.CountryDocumentID = source.CountryDocumentID
            AND target.CountryID = source.CountryID
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
                INSERT (CountryDocumentID, CountryID, PillarID, SectionPath,
                        SectionTitle, SectionLevel, PageStart, PageEnd)
                VALUES (source.CountryDocumentID, source.CountryID, source.PillarID,
                        source.SectionPath, source.SectionTitle,
                        source.SectionLevel, source.PageStart, source.PageEnd)

            OUTPUT inserted.TOCID;
            """

        params = (
            country_doc_id,
            country_id,
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
        country_doc_id: int,
        country_id: Optional[int],
        pillar_id: Optional[int],
    ) -> None:
        if not chunks:
            return

        query = """
            INSERT INTO DocumentChunks
                (ChunkID, CountryDocumentID, TOCID, CountryID, PillarID,
                 ChunkIndex, ChunkText)
            VALUES (?,?,?,?,?,?,?)
        """
        params = [
            (
                c.get("chunk_id"),
                country_doc_id,
                c.get("toc_id"),
                country_id,
                pillar_id,
                c.get("chunk_index"),
                c.get("chunk_text"),
            )
            for c in chunks
        ]

        await self.engine.execute_write_async(query, params, executemany=True)

    def test_connection(self) -> bool:
       return self.engine.test_connection()

    async def get_active_pillars(self) -> List[Dict[str, Any]]:
        """Return active pillars ordered for AI prompt construction."""
        query = """
            SELECT PillarID, PillarName, Description, DisplayOrder
            FROM Pillars
            WHERE IsDeleted = 0 AND IsActive = 1
            ORDER BY DisplayOrder, PillarID
        """
        return await self.engine.fetch_dicts_async(query)

    async def get_active_pillars_map(self) -> Dict[int, Dict[str, Any]]:
        pillars = await self.get_active_pillars()
        return {int(p["PillarID"]): p for p in pillars}

    async def get_active_countries(self) -> List[Dict[str, Any]]:
        """Active countries for GDELT scope and emerging-trends context."""
        query = """
            SELECT CountryName, Region, CountryCode
            FROM Countries
            WHERE IsDeleted = 0
            ORDER BY Region, CountryName
        """
        return await self.engine.fetch_dicts_async(query)

    async def get_ai_country_context(
    self,
        country_id: int,
        year: int,
        pillar_id: Optional[int] = None,
    ) -> Dict[str, Any]:

        query = """
            SELECT 
                a.AIProgress as AfricaHealthScore,
                c.CountryName,
                c.Continent,
                a.EvidenceSummary,
                a.StructuralEvidence,
                a.OutcomeEvidence,
                a.PerceptionEvidence,
                a.CrossPillarPatterns,
                a.StrategicRecommendation,
                p.PillarName
            FROM AICountryScores a
            JOIN Countries c 
                ON a.CountryID = c.CountryID 
                AND c.IsDeleted = 0
            left join pillars p on p.PillarID=?
            WHERE a.CountryID = ?
            AND a.Year = ?
        """

        params = (pillar_id,country_id, year)

        result = await self.engine.fetch_dicts_async(query, params)

        return result[0] if result else None
        
    async def save_immediate_situation_summary(
        self,
        country_id: int,
        year: int,
        record: dict
    ) -> None:

        if not record:
            return

        query = """
            UPDATE AICountryScores
            SET 
                ImmediateSituationSummary = ?,
                KeyDevelopments = ?,
                CriticalRisks = ?,
                Gaps = ?,
                KeyFindings = ?,
                Recommendations = ?,
                EvidenceSummary = CASE 
                    WHEN ? IS NOT NULL AND LTRIM(RTRIM(CAST(? AS NVARCHAR(MAX)))) <> '' 
                    THEN ? 
                    ELSE EvidenceSummary 
                END
            WHERE CountryID = ?
            AND Year = ?
        """

        exec_summary = record.get("executive_summary")

        params = (
            record.get("immediateSituationSummary"),
            record.get("key_developments"),
            record.get("critical_risks"),
            record.get("gaps"),
            record.get("key_findings"),
            record.get("recommendations"),
            exec_summary,   # check NULL
            exec_summary,   # check empty
            exec_summary,   # value to update
            country_id,
            year
        )

        await self.engine.execute_write_async(query, params)


    async def get_FAQ_context(self, isglobal: bool = False) -> List[Dict]:

        where = "WHERE Related LIKE 'global'" if isglobal else "WHERE Related LIKE 'country'"

        query = f"""
            SELECT FAQID, Related, Category, QuestionText
            FROM AIAssistantFAQ
            {where}
        """

        return await self.engine.fetch_dicts_async(query)

    async def GetLocalContextDataForLLM(self, FAQIDs: List[str],country_id: Optional[int] = None,  pillarId: Optional[int] = None) -> List[Dict]:

        query = """
            EXEC dbo.usp_GetLocalContextDataForLLM ?, ?, ?
        """

        params = (
            json.dumps(FAQIDs),country_id, pillarId
        )
        response = await self.engine.fetch_dicts_async(query, params)

        return response
    
    async def GetCrossComparisionLocalContextDataForLLM(self, country_ids: List[str]) -> List[Dict]:

        query = """
            EXEC dbo.usp_CountryCrossComparision_faq ?
        """

        params = (
            json.dumps(country_ids)
        )
        response = await self.engine.fetch_dicts_async(query, params)

        return response



# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

db_repository = DatabaseRepository()