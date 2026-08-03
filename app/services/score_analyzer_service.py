"""
Score Analyzer Service
----------------------
Orchestrates AI scoring for questions, pillars, and countries.
Delegates all LLM calls to AHIResearchService.
Persists results via DatabaseRepository.
"""

import math
import logging
from datetime import datetime
from typing import Any, Optional
logger = logging.getLogger(__name__)
from app.services.core.repository import DatabaseRepository
from app.services.common.ahi_ai_research_service import AHIResearchService
from app.services.common.json_response_parser import (
    normalize_numbered_list_text,
    apply_data_sourcing_to_sources,
)
from app.services.rag_query_service import rag_query_service
#  To DB after every N records (currently 1 = immediate upsert).
#  Increase for bulk jobs to reduce round-trips.
from app.services.common.realtime_operational_stress_prompt import ROSEW_PILLAR_ID

_BATCH_SIZE = 5


# =========================================================================== #
class ScoreAnalyzerService:
    """
    Coordinates AI scoring workflows across questions, pillars, and countries.

    Responsibilities
    ----------------
    - Fetch evaluation data from DB views
    - Call AHIResearchService for AI scoring
    - Build DB-ready records
    - Upsert results in configurable batches
    """

    __slots__ = ("_db", "_ai")

    def __init__(self) -> None:
        self._db = DatabaseRepository()
        self._ai = AHIResearchService()

    # ------------------------------------------------------------------ #
    #  Safe type converters                                              #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _to_float(value) -> float:
        """Convert any value to a finite float, defaulting to 0.0."""
        if value is None:
            return 0.0
        if isinstance(value, float):
            return 0.0 if (math.isnan(value) or math.isinf(value)) else round(value, 2)
        if isinstance(value, int):
            return float(value)
        if isinstance(value, str):
            s = value.strip().lower()
            if s in {"", "null", "none", "nan", "inf", "-inf", "infinity", "-infinity"}:
                return 0.0
            try:
                val = float(s.replace(",", ""))
                return 0.0 if (math.isnan(val) or math.isinf(val)) else round(val, 2)
            except (ValueError, TypeError):
                return 0.0
        return 0.0


    @staticmethod
    def _to_float_or_none(value):
        if value is None:
            return None

        try:
            val = float(str(value).replace(",", "").strip())

            if math.isnan(val) or math.isinf(val):
                return 0.0

            return round(val, 2)
        except (ValueError, TypeError):
            return None
    
    @staticmethod
    def _to_int(value) -> int:
        """Convert any value to an int, defaulting to 0."""
        if value is None:
            return 0
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return 0 if (math.isnan(value) or math.isinf(value)) else int(value)
        if isinstance(value, str):
            s = value.strip().lower()
            if s in {"", "null", "none", "nan", "inf", "-inf", "infinity", "-infinity"}:
                return 0
            try:
                return int(float(s.replace(",", "")))
            except (ValueError, TypeError):
                return 0
        return 0

    @staticmethod
    def _discrepancy(ai_progress: float, evaluator_score: Optional[float]) -> float:
        """Absolute difference between AI and evaluator scores."""
        if evaluator_score is not None:
            return abs(ai_progress - evaluator_score)
        return ai_progress

    # ------------------------------------------------------------------ #
    #  Data fetch helpers                                                 #
    # ------------------------------------------------------------------ #

    async def _fetch_countries(self, country_id: Optional[int] = None):
        where = (
            f"WHERE IsDeleted = 0 AND CountryID = {country_id}"
            if country_id
            else "WHERE IsDeleted = 0"
        )
        return await self._db.engine.fetch_df_async(
            f"SELECT CountryID, CountryName, Continent FROM Countries {where}"
        )

    @staticmethod
    def _continent_label(country) -> str:
        return f"Continent: {country.Continent}, Country: {country.CountryName}"

    # ------------------------------------------------------------------ #
    #  Public entry points                                                #
    # ------------------------------------------------------------------ #

    async def analyze_all_countries(self, country_id: Optional[int] = None) -> bool:
        """
        Run full analysis (questions → pillars → country) for all countries,
        or a single country when country_id is provided.
        """
        try:
            countries = await self._fetch_countries(country_id)
            if countries.empty:
                logger.error("No countries found for analysis.")
                return False

            for country in countries.itertuples(index=False):
                try:
                    await self._analyze_questions(country)
                    await self._analyze_pillars(country)
                    await self._analyze_country(country)
                except Exception as exc:
                    logger.error(
                        "Country %d (%s) analysis failed: %s",
                        country.CountryID,
                        country.CountryName,
                        exc,
                    )
            return True

        except Exception as exc:
            logger.error("analyze_all_countries failed: %s", exc, exc_info=True)
            raise

    async def analyze_single_country(self, country_id: int) -> bool:
        """Score overall country-level assessment only."""
        return await self._run_for_country(country_id, self._analyze_country)

    async def analyze_country_pillars(
        self, country_id: int, pillar_id: Optional[int] = None
    ) -> bool:
        """Score all pillars (or a single pillar) for a country."""
        return await self._run_for_country(
            country_id, self._analyze_pillars, pillar_id=pillar_id
        )

    async def analyze_country_questions(
        self, country_id: int, pillar_id: Optional[int] = None
    ) -> bool:
        """Score all questions (or a single pillar's questions) for a country."""
        return await self._run_for_country(
            country_id, self._analyze_questions, pillar_id=pillar_id
        )
    
    async def import_missing_country_questions(
        self, country_id: int, pillar_id: Optional[int] = None
    ) -> bool:
        """Score all questions (or a single pillar's questions) for a country."""
        return await self._run_for_country(
            country_id, self._analyze_questions, pillar_id=pillar_id,missing_only=True
        )
    # ------------------------------------------------------------------ #
    #  Internal dispatcher                                                #
    # ------------------------------------------------------------------ #

    async def _run_for_country(self, country_id: int, handler, **kwargs) -> bool:
        """Fetch a single country row then call *handler* on it."""
        try:
            countries = await self._fetch_countries(country_id)
            if countries.empty:
                return False
            for country in countries.itertuples(index=False):
                await handler(country, **kwargs)
            return True
        except Exception as exc:
            logger.error(
                "%s failed for country %d: %s",
                handler.__name__,
                country_id,
                exc,
                exc_info=True,
            )
            raise

    # ------------------------------------------------------------------ #
    #  Core analyzers                                                     #
    # ------------------------------------------------------------------ #

    async def _analyze_questions(
        self,
        country: Any,
        pillar_id: Optional[int] = None,
        missing_only = False
    ) -> bool:

        countryID = int(country.CountryID)
        year = datetime.now().year

        where = f"CountryID = {countryID}"

        if pillar_id is not None:
            where += f" AND PillarID = {pillar_id}"

        if missing_only:
            where += f"""
                AND QuestionID NOT IN
                (
                    SELECT QuestionID
                    FROM AIEstimatedQuestionScores
                    WHERE Year = {year} AND {where}
                )
            """

        df = await self._db.get_view_data("vw_AiCountryPillarQuestionEvaluations", where)
        if df.empty:
            logger.info("No questions found: country %d", country.CountryID)
            return False

        target_pillars = (
            [pillar_id] if pillar_id is not None else df["PillarID"].unique().tolist()
        )

        for pid in target_pillars:
            batch: list[dict] = []

            for row in df[df["PillarID"] == pid].itertuples(index=False):
                try:
                    ai_data = await self._ai.research_and_score_question(
                        country_name=country.CountryName,
                        continent=self._continent_label(country),
                        pillarID=row.PillarID,
                        pillar_name=row.PillarName,
                        question_text=row.QuestionText,
                    )
                    if not ai_data.get("success"):
                        logger.warning(
                            "AI failed for question %d, country %d",
                            row.QuestionID,
                            country.CountryID,
                        )
                        continue

                    normalized = self._safe_normalized(row.NormalizedValue)
                    if int(row.PillarID) == ROSEW_PILLAR_ID:
                        if ai_data["AIScore"] is not None and int(ai_data["AIScore"]) >= 0:
                            batch.append(self._build_question_record(row, ai_data, normalized))
                    else:
                        batch.append(self._build_question_record(row, ai_data, normalized))
                    batch = await self._flushQuestion(
                        country.CountryID, batch ,self._db.bulk_upsert_question_evaluations
                    )

                except Exception as exc:
                    logger.error(
                        "Question %d, country %d: %s",
                        row.QuestionID,
                        country.CountryID,
                        exc,
                        exc_info=True,
                    )

            await self._flushQuestion(
                country.CountryID, batch ,self._db.bulk_upsert_question_evaluations, force=True
            )
            await self._db.AiInsertAnalyticalLayerResults(country.CountryID)

        return True

    async def _analyze_pillars(
        self,
        country: Any,
        pillar_id: Optional[int] = None,
    ) -> bool:
        """Score every pillar for a country."""
        where = f"countryId = {country.CountryID}"
        if pillar_id is not None:
            where += f" AND PillarID = {pillar_id}"

        df = await self._db.get_view_data("vw_AiCountryPillarEvaluation", where)
        if df.empty:
            logger.info("No pillar evaluations found: country %d", country.CountryID)
            return False

        pillar_batch: list[dict] = []
        source_batch: list[dict] = []

        for row in df.itertuples(index=False):
            try:
                ai_data = await self._ai.research_and_score_pillar(
                    country_name=country.CountryName,
                    continent=self._continent_label(country),
                    pillarId=row.PillarID,
                    pillar_name=row.PillarName
                )
                if not ai_data.get("success"):
                    continue

                pillar_batch.append(
                    self._build_pillar_record(row, ai_data, country.CountryID)
                )
                source_batch.extend(self._build_source_records(row, ai_data))

                pillar_batch, source_batch = await self._flush_pillar(
                    pillar_batch, source_batch
                )

            except Exception as exc:
                logger.error(
                    "Pillar %d, country %d: %s",
                    row.PillarID,
                    country.CountryID,
                    exc,
                    exc_info=True,
                )

        await self._flush_pillar(pillar_batch, source_batch, force=True)
        await self._db.AiRecalculateCountryScore(country.CountryID)
        
        return True

    async def _analyze_country(self, country: Any, **_) -> bool:
        """Score the overall country-level Healthassessment."""
        df = await self._db.get_view_data(
            "vw_AiCountryEvaluations", f"countryId = {country.CountryID}"
        )
        if df.empty:
            logger.info("No country evaluations found: country %d", country.CountryID)
            return False

        batch: list[dict] = []

        for row in df.itertuples(index=False):
            try:
                ai_data = await self._ai.research_and_score_country(
                    country_name=country.CountryName,
                    continent=self._continent_label(country),
                )
                if not ai_data.get("success"):
                    continue

                batch.append(self._build_country_record(row, ai_data))
                batch = await self._flush(
                    batch, self._db.bulk_upsert_country_evaluations
                )

            except Exception as exc:
                logger.error(
                    "Country evaluation %d: %s",
                    country.CountryID,
                    exc,
                    exc_info=True,
                )

        await self._flush(batch, self._db.bulk_upsert_country_evaluations, force=True)

        await self.immediateSituation(country.CountryID)
        await self._db.AiRecalculateCountryScore(country.CountryID)
        
        return True

    async def immediateSituation(self, country_id: int, **_) -> bool:
        """Score the overall country-level Healthassessment."""
        year = datetime.now().year        

        ai_country= await self._db.get_ai_country_context(country_id, year)
        country_Name = ai_country["CountryName"]
        continent =ai_country["Continent"]

        question = f"""
        What are the most critical recent developments, emerging risks, structural weaknesses, and key strengths across all major sectors in {country_Name}? Include insights on governance, security, economy, social cohesion, infrastructure, and institutional effectiveness. Focus on cross-pillar patterns and high-impact information relevant for executive-level country assessment and situational awareness.
        """

        document_context = await rag_query_service.get_country_document_context(country_id, question)

        if ai_country:
            ai_country_context = "\n".join(f"{key}: {value}" for key, value in ai_country.items())
        else:
            ai_country_context = ""

        ai_data = await self._ai.immediate_situation(
                    country_name=country_Name,
                    continent =continent,
                    ai_country_context=ai_country_context,
                    documentContext=document_context,
                    year=year
                )

        result = self._build_immediateSituation_record(country_id, ai_data)
        
        await self._db.save_immediate_situation_summary(country_id,year,result)
        
        
        return True

    # ------------------------------------------------------------------ #
    #  Record builders                                                   #
    # ------------------------------------------------------------------ #

    def _build_question_record(
        self,
        row: Any,
        ai: dict,
        normalized_value: float,
    ) -> dict:
        ai_progress = self._to_float(ai.get("AIProgress") or 0)
        evaluator_score = self._to_float(normalized_value * 100)

        return {
            "CountryID": row.CountryID,
            "PillarID": row.PillarID,
            "QuestionID": row.QuestionID,
            "Year": self._to_int(ai.get("Year")),
            "AIScore": self._to_float_or_none(ai.get("AIScore")),
            "AIProgress": ai_progress,
            "EvaluatorScore": evaluator_score,
            "Discrepancy": self._discrepancy(ai_progress, evaluator_score),
            "ConfidenceLevel": ai.get("ConfidenceLevel"),
            "EvidenceSummary": ai.get("EvidenceSummary"),
            "StructuralEvidence": ai.get("StructuralEvidence"),
            "OperationalEvidence": ai.get("OperationalEvidence"),
            "OutcomeEvidence": ai.get("OutcomeEvidence"),
            "PerceptionEvidence": ai.get("PerceptionEvidence"),
            "TemporalScope": ai.get("TemporalScope"),
            "DistortionScreening": ai.get("DistortionScreening"),
            "RelationalDependencies": ai.get("RelationalDependencies"),
            "StressPoliticalShock": ai.get("StressPoliticalShock"),
            "StressEconomicShock": ai.get("StressEconomicShock"),
            "StressNarrativeShock": ai.get("StressNarrativeShock"),
            "StressOverallResilienceShock": ai.get("StressOverallResilienceShock"),
            "InequalityAdjustment": ai.get("InequalityAdjustment"),
            "OpacityRisk": ai.get("OpacityRisk"),
            "RedFlag": ai.get("RedFlag"),
            "SourceName": ai.get("SourceName"),
            "SourceType": ai.get("SourceType"),
            "SourceURL": ai.get("SourceURL"),
            "SourceDataYear": self._to_int(ai.get("SourceDataYear")),
            "SourceHierarchyLevel": self._to_int(ai.get("SourceHierarchyLevel")),
            "SourceDataExtract": ai.get("SourceDataExtract"),
            "SourcesConsulted": self._to_int(ai.get("SourcesConsulted")),
        }

    def _build_pillar_record(self, row: Any, ai: dict, country_id: int) -> dict:
        ai_progress = self._to_float(ai.get("AIProgress") or 0)
        evaluator_score = self._to_float(row.EvaluatorScore)

        return {
            "CountryID": country_id,
            "PillarID": row.PillarID,
            "Year": ai.get("Year"),
            "AIScore": self._to_float(ai.get("AIScore")),
            "AIProgress": ai_progress,
            "EvaluatorScore": evaluator_score,
            "Discrepancy": self._discrepancy(ai_progress, evaluator_score),
            "ConfidenceLevel": ai.get("ConfidenceLevel"),
            "EvidenceSummary": ai.get("EvidenceSummary"),
            "StructuralEvidence": ai.get("StructuralEvidence"),
            "OperationalEvidence": ai.get("OperationalEvidence"),
            "OutcomeEvidence": ai.get("OutcomeEvidence"),
            "PerceptionEvidence": ai.get("PerceptionEvidence"),
            "TemporalScope": ai.get("TemporalScope"),
            "DistortionScreening": ai.get("DistortionScreening"),
            "RelationalIntegrity": ai.get("RelationalIntegrity"),
            "StressPoliticalShock": ai.get("StressPoliticalShock"),
            "StressEconomicShock": ai.get("StressEconomicShock"),
            "StressNarrativeShock": ai.get("StressNarrativeShock"),
            "StressOverallResilience": ai.get("StressOverallResilience"),
            "StressScoreAdjustment": ai.get("StressScoreAdjustment"),
            "InequalityAdjustment": ai.get("InequalityAdjustment"),
            "OpacityRisk": ai.get("OpacityRisk"),
            "NonCompensationNote": ai.get("NonCompensationNote"),
            "GeographicEquityNote": ai.get("GeographicEquityNote"),
            "InstitutionalAssessment": ai.get("InstitutionalAssessment"),
            "DataGapAnalysis": ai.get("DataGapAnalysis"),
            "RedFlag": ai.get("RedFlag"),
        }

    def _build_country_record(self, row: Any, ai: dict) -> dict:
        ai_progress = self._to_float(ai.get("AIProgress") or 0)
        evaluator_score = self._to_float(row.EvaluatorScore)

        return {
            "CountryID": row.CountryID,
            "Year": self._to_int(ai.get("Year") or datetime.now().year),
            "AIScore": self._to_float(ai.get("AIScore")),
            "AIProgress": ai_progress,
            "EvaluatorScore": evaluator_score,
            "Discrepancy": self._discrepancy(ai_progress, evaluator_score),
            "ConfidenceLevel": ai.get("ConfidenceLevel", "Unknown"),
            "EvidenceSummary": ai.get("ExecutiveSummary"),
            "StructuralEvidence": ai.get("StructuralEvidence"),
            "OperationalEvidence": ai.get("OperationalEvidence"),
            "OutcomeEvidence": ai.get("OutcomeEvidence"),
            "PerceptionEvidence": ai.get("PerceptionEvidence"),
            "TemporalScope": ai.get("TemporalScope"),
            "DistortionScreening": ai.get("DistortionScreening"),
            "PoliticalShock": ai.get("PoliticalShock"),
            "EconomicShock": ai.get("EconomicShock"),
            "NarrativeShock": ai.get("NarrativeShock"),
            "OverallStressResilience": ai.get("OverallStressResilience"),
            "StressScoreAdjustment": ai.get("StressScoreAdjustment"),
            "InequalityAdjustment": ai.get("InequalityAdjustment"),
            "OpacityRisk": ai.get("OpacityRisk"),
            "NonCompensationNote": ai.get("NonCompensationNote"),
            "CrossPillarPatterns": ai.get("CrossPillarPatterns"),
            "RelationalIntegrity": ai.get("RelationalIntegrity"),
            "InstitutionalCapacity": ai.get("InstitutionalCapacity"),
            "EquityAssessment": ai.get("EquityAssessment"),
            "ConflictRiskOutlook": ai.get("ConflictRiskOutlook"),
            "StrategicRecommendation": ai.get("StrategicRecommendation"),
            "DataTransparencyNote": ai.get("DataTransparencyNote"),
            "PrimarySource": ai.get("PrimarySource"),
            "VerifiedBy": None,
        }

    def _build_source_records(self, row: Any, ai: dict) -> list[dict]:
        """Expand the Sources list from a pillar AI response into flat DB records."""
        records: list[dict] = []
        target_year = self._to_int(ai.get("Year")) or datetime.now().year
        sources = apply_data_sourcing_to_sources(ai.get("Sources", []), target_year)

        for src in sources:
            extract = (src.get("data_extract") or "").strip()
            quality_flag = (src.get("data_quality_flag") or "").strip()
            reporting_lag = src.get("reporting_lag")

            # Single compact metadata prefix (values already recomputed vs Target Year)
            meta_parts: list[str] = []
            if quality_flag and quality_flag != "Current":
                meta_parts.append(f"[{quality_flag}]")
            if (
                reporting_lag is not None
                and str(reporting_lag).strip() != ""
                and int(reporting_lag) > 0
            ):
                meta_parts.append(f"[Reporting Lag: {reporting_lag}]")
            if meta_parts and not extract.startswith("["):
                extract = f"{' '.join(meta_parts)} {extract}".strip()

            records.append(
                {
                    "CountryID": row.CountryID,
                    "PillarID": row.PillarID,
                    "DataYear": self._to_int(src.get("data_year")),
                    "SourceType": src.get("source_type"),
                    "SourceName": src.get("source_name"),
                    "SourceURL": src.get("source_url"),
                    "DataExtract": extract,
                    "TrustLevel": self._to_int(src.get("source_trust_level")),
                }
            )
        return records



    def _build_immediateSituation_record(self, countryId: int, ai: dict) -> dict:
        summary = ai.get("executive_summary", "")

        return {
            "CountryID": countryId,
            "immediateSituationSummary": ai.get("immediateSituationSummary", "Unknown"),
            "key_developments": normalize_numbered_list_text(ai.get("key_developments", "Unknown")),
            "critical_risks": normalize_numbered_list_text(ai.get("critical_risks")),
            "gaps": normalize_numbered_list_text(ai.get("gaps")),
            "key_findings": normalize_numbered_list_text(ai.get("key_findings")),
            "recommendations": normalize_numbered_list_text(ai.get("recommendations")),
            "executive_summary": summary if isinstance(summary, str) and len(summary) > 50 else ""
        }

    # ------------------------------------------------------------------ #
    #  Batch flush helpers                                               #
    # ------------------------------------------------------------------ #

    async def _flushQuestion(
        self,
        countryID:int,
        batch: list[dict],
        upsert_fn,
        *,
        force: bool = False,
    ) -> list[dict]:
        """
        Upsert *batch* when it reaches _BATCH_SIZE (or when force=True).
        Returns an empty list after flushing, or the original list if not yet full.
        """
        if batch and (force or len(batch) >= _BATCH_SIZE):
            await upsert_fn(batch,countryID)
            return []
        return batch
    
    async def _flush(
        self,
        batch: list[dict],
        upsert_fn,
        *,
        force: bool = False,
    ) -> list[dict]:
        """
        Upsert *batch* when it reaches _BATCH_SIZE (or when force=True).
        Returns an empty list after flushing, or the original list if not yet full.
        """
        if batch and (force or len(batch) >= _BATCH_SIZE):
            await upsert_fn(batch)
            return []
        return batch

    async def _flush_pillar(
        self,
        pillar_batch: list[dict],
        source_batch: list[dict],
        *,
        force: bool = False,
    ) -> tuple[list[dict], list[dict]]:
        """Paired flush for pillar records + their source records."""
        if pillar_batch and (force or len(pillar_batch) >= _BATCH_SIZE):
            await self._db.bulk_upsert_pillar_evaluations(pillar_batch, source_batch)
            return [], []
        return pillar_batch, source_batch

    # ------------------------------------------------------------------ #
    #  Utility                                                            #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _safe_normalized(value) -> float:
        """Return 0.0 if NormalizedValue is None or NaN, otherwise the value."""
        if value is None:
            return 0.0
        if isinstance(value, float) and math.isnan(value):
            return 0.0
        return float(value)


# Module-level singleton
score_analyzer_service = ScoreAnalyzerService()
