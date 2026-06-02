"""
Score analyzer service - LLM-powered analysis with database exception logging
"""

from datetime import datetime
import math
import logging
from typing import Any, Optional
from app.services.core.repository import DatabaseRepository
from app.services.common.veridian_ai_research_service import VerdianAIResearchService
from app.services.rag_query_service import rag_query_service

logger = logging.getLogger(__name__)
_BATCH_SIZE = 5

class ScoreAnalyzerService:
    """Service for analyzing SQL Server data using LLM"""

    __slots__ = ('db_service', '_ai')  # Memory optimization

    def __init__(self):
        self.db_service = DatabaseRepository()
        self._ai = VerdianAIResearchService()

    @staticmethod
    def to_float_safe(value) -> float:
        """Convert value to float safely, returning 0.0 for invalid values"""
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
    def to_float_none(value) -> float | None:
        """Convert value to float safely. Returns None for invalid values."""

        if value is None:
            return None

        try:
            if isinstance(value, str):
                s = value.strip().lower()

                if s in {"", "null", "none", "nan", "inf", "-inf", "infinity", "-infinity"}:
                    return None

                value = float(s.replace(",", ""))

            val = float(value)

            if math.isnan(val) or math.isinf(val):
                return None

            return round(val, 2)

        except (ValueError, TypeError):
            return None
    
    @staticmethod
    def to_int_safe(value) -> int:
        """Convert value to int safely, returning 0 for invalid values"""
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
    
    async def _get_city_data(self, city_id: Optional[int] = None):
        where = (
            f"WHERE IsDeleted = 0 AND CityID = {city_id}"
            if city_id
            else "WHERE IsDeleted = 0"
        )
        return await self.db_service.engine.fetch_df_async(
            f"Select CityID, CityName, State, Country from Cities {where}"
        )


    async def analyze_all_cities_questions(self, city_id: Optional[int] = None) -> bool:
        """Analyze City Questions data for all cities or specific city"""
        try:
            df = await self._get_city_data(city_id)

            if df.empty:
                logger.error("No cities found for analysis analyze_all_cities_questions endpoint")
                return False

            for city in df.itertuples(index=False):
                try:
                    await self.analyze_PillarQuestions(city)
                    await self.analyze_cityPillar(city)
                    await self.analyze_city(city)
                except Exception as e:
                    logger.error(f"Failed to analyze city {city.CityID} ({city.CityName}): {e}")
                    continue

            return True
            
        except Exception as e:
            logger.error(f"Error in analyze_all_cities_questions: {e}")
            raise

    async def analyze_single_City(self, cityId: int) -> bool:
        """Analyze City Questions data for a specific city"""
        try:
            df = await self._get_city_data(cityId)
            if df.empty:
                return False

            for city in df.itertuples(index=False):
                await self.analyze_city(city)

            return True
            
        except Exception as e:
            logger.error(f"Error in analyze_single_City (CityID: {cityId}): {e}")
            raise

    async def analyze_city_pillars(self, cityId: int) -> bool:
        """Analyze City pillar data for a specific city"""
        try:
            df = await self._get_city_data(cityId)
            if df.empty:
                return False

            for city in df.itertuples(index=False):
                await self.analyze_cityPillar(city)

            return True
            
        except Exception as e:
            logger.error(f"Error in analyze_city_pillars (CityID: {cityId}): {e}")
            raise

    async def analyze_Single_Pillar(self, cityId: int, pillar_id: Optional[int] = None) -> bool:
        """Analyze specific pillar for a city"""
        try:
            df = await self._get_city_data(cityId)
            if df.empty:
                return False

            for city in df.itertuples(index=False):
                await self.analyze_cityPillar(city, pillar_id)

            return True
            
        except Exception as e:
            logger.error(f"Error in analyze_Single_Pillar (CityID: {cityId}, PillarID: {pillar_id}): {e}")
            raise

    async def analyze_questions_of_city_pillar(self, cityId: int, pillar_id: Optional[int] = None) -> bool:
        """Analyze questions for city pillar"""
        try:
            df = await self._get_city_data(cityId)
            if df.empty:
                return False

            for city in df.itertuples(index=False):
                await self.analyze_PillarQuestions(city, pillar_id)

            return True
            
        except Exception as e:
            logger.error(f"Error in analyze_questions_of_city_pillar (CityID: {cityId}): {e}")
            raise

    def _build_question_record(self, row, ai_data, normalized_value: float) -> dict[str, Any]:
        """Build question evaluation record from AI data"""

        return {
            "CityID": row.CityID,
            "PillarID": row.PillarID,
            "QuestionID": row.QuestionID,
            "Year": self.to_int_safe(ai_data["Year"]),
            "AIScore": self.to_float_none(ai_data["AIScore"]),
            "AIProgress": self.to_float_safe(ai_data["AIProgress"]),
            "EvaluatorProgress": self.to_float_safe(normalized_value * 100),
            "Discrepancy": self.to_float_safe(ai_data["Discrepancy"]),
            "ConfidenceLevel": ai_data["ConfidenceLevel"],
            "DataSourcesUsed": self.to_int_safe(ai_data["DataSourcesCount"]),
            "EvidenceSummary": ai_data["EvidenceSummary"],
            "RedFlags": ai_data["RedFlag"],
            "GeographicEquityNote": ai_data["GeographicEquityNote"],
            "SourceType": ai_data["SourceType"],
            "SourceName": ai_data["SourceName"],
            "SourceURL": ai_data["SourceURL"],
            "SourceDataYear": self.to_int_safe(ai_data["SourceDataYear"]),
            "SourceDataExtract": ai_data["SourceDataExtract"],
            "SourceTrustLevel": self.to_int_safe(ai_data["SourceHierarchyLevel"])
        }

    async def analyze_PillarQuestions(
    self,
    city: Any,
    pillar_id: Optional[int] = None,
    missing_only: bool = False
) -> bool:
        """Analyze pillar questions data for a city."""

        year = datetime.now().year
        
        city_id = int(city.CityID)

        where = f"CityID = {city_id}"

        if pillar_id is not None:
            where += f" AND PillarID = {pillar_id}"


        if missing_only:           

            where += f"""
                AND QuestionID NOT In
                (
                    select QuestionID from AIEstimatedQuestionScores where Year = {year}
                )
            """

        df = await self.db_service.get_view_data(
            "vw_AiCityPillarQuestionEvaluations",
            where,
        )

        if df.empty:
            logger.info(
                "No pillar questions found: city %d (%s)",
                city_id,
                city.CityName,
            )
            return False

        target_pillars = (
            [pillar_id]
            if pillar_id is not None
            else df["PillarID"].unique().tolist()
        )

        for pid in target_pillars:

            batch: list[dict[str, Any]] = []

            for row in df[df["PillarID"] == pid].itertuples(index=False):

                try:

                    normalized_value = self._safe_normalized(
                        row.NormalizedValue
                    )

                    ai_data = await self._ai.research_and_score_question(
                        city.CityName,
                        f"State :{city.State}, Country :{city.Country}",
                        row.PillarID,
                        row.PillarName,
                        f" Question :{row.QuestionText}, Options :{row.Options}",
                        row.ScoreProgress,
                        round(normalized_value * 4.0),
                        None,
                    )

                    if not ai_data.get("success"):

                        logger.warning(
                            "AI analysis failed for question %d in city %d",
                            row.QuestionID,
                            city_id,
                        )

                        continue

                    batch.append(
                        self._build_question_record(
                            row,
                            ai_data,
                            normalized_value,
                        )
                    )

                    batch = await self._flushQuestion(
                        city_id,
                        batch,
                        self.db_service.bulk_upsert_question_evaluations,
                    )

                except Exception as exc:

                    logger.error(
                        "Question %d, city %d: %s",
                        row.QuestionID,
                        city_id,
                        exc,
                        exc_info=True,
                    )

            await self._flushQuestion(
                city_id,
                batch,
                self.db_service.bulk_upsert_question_evaluations,
                force=True,
            )

            await self.db_service.AiInsertAnalyticalLayerResults(
                city_id
            )

        return True    


    async def analyze_cityPillar( self, city: Any, pillar_id: Optional[int] = None,) -> bool:
        """Score every pillar for a city."""

        where = f"cityId = {city.CityID}"
        if pillar_id is not None:
            where += f" AND PillarID = {pillar_id}"

        df = await self.db_service.get_view_data(
            "vw_AiCityPillarEvaluation",
            where,
        )

        if df.empty:
            logger.info(
                "No pillar evaluations found: city %d",
                city.CityID,
            )
            return False

        pillar_batch: list[dict[str, Any]] = []
        source_batch: list[dict[str, Any]] = []

        for row in df.itertuples(index=False):
            try:
                ai_data = await self._ai.research_and_score_pillar(
                    city.CityName,
                    f"State :{city.State}, Country :{city.Country}",
                    row.PillarID,
                    row.PillarName,
                    row.QuestionWithScores,
                    row.EvaluatorProgress,
                    row.AIScore,
                )

                if not ai_data.get("success"):
                    logger.warning(
                        "AI analysis failed for pillar %d in city %d",
                        row.PillarID,
                        city.CityID,
                    )
                    continue

                pillar_batch.append(
                    self._build_pillar_record(
                        row,
                        ai_data,
                        city.CityID,
                    )
                )

                source_batch.extend(
                    self._build_source_records(
                        row,
                        ai_data,
                    )
                )

                pillar_batch, source_batch = await self._flush_pillar(
                    pillar_batch,
                    source_batch,
                )

            except Exception as exc:
                logger.error(
                    "Pillar %d, city %d: %s",
                    row.PillarID,
                    city.CityID,
                    exc,
                    exc_info=True,
                )

        await self._flush_pillar(
            pillar_batch,
            source_batch,
            force=True,
        )

        await self.db_service.AiRecalculateCityScore(
            city.CityID
        )

        return True

    async def analyze_city(self, city: Any) -> bool:
        """Analyze overall city data and generate comprehensive evaluation."""
        
        df = await self.db_service.get_view_data(
            "vw_AiCityEvaluations",
            f"cityId = {city.CityID}"
        )

        if df.empty:
            logger.info(
                "No city evaluations found: city %d (%s)",
                city.CityID,
                city.CityName,
            )
            return False

        batch: list[dict[str, Any]] = []

        for row in df.itertuples(index=False):
            try:
                ai_data = await self._ai.research_and_score_city(
                    city.CityName,
                    f"State :{city.State}, Country :{city.Country}",
                    row.EvaluatorProgress,
                    row.AIScore,
                    row.PillarWithScores,
                )

                if not ai_data.get("success"):
                    logger.warning(
                        "AI analysis failed for city %d",
                        city.CityID,
                    )
                    continue

                batch.append({
                    "CityID": row.CityID,
                    "Year": self.to_int_safe(ai_data.get("Year")),
                    "AIScore": self.to_float_safe(ai_data.get("AIScore")),
                    "AIProgress": self.to_float_safe(ai_data.get("AIProgress")),
                    "EvaluatorProgress": self.to_float_safe(row.EvaluatorProgress),
                    "Discrepancy": self.to_float_safe(ai_data.get("Discrepancy")),
                    "ConfidenceLevel": ai_data.get("ConfidenceLevel"),
                    "EvidenceSummary": ai_data.get("EvidenceSummary"),
                    "CrossPillarPatterns": ai_data.get("CrossPillarPatterns", ""),
                    "InstitutionalCapacity": ai_data.get("InstitutionalCapacity"),
                    "EquityAssessment": ai_data.get("EquityAssessment"),
                    "SustainabilityOutlook": ai_data.get("SustainabilityOutlook"),
                    "StrategicRecommendations": ai_data.get("StrategicRecommendation"),
                    "DataTransparencyNote": ai_data.get("DataTransparencyNote"),
                })

                batch = await self._flush(
                    batch,
                    self.db_service.bulk_upsert_city_evaluations,
                )

            except Exception as exc:
                logger.error(
                    "City evaluation %d: %s",
                    city.CityID,
                    exc,
                    exc_info=True,
                )

        await self._flush(
            batch,
            self.db_service.bulk_upsert_city_evaluations,
            force=True,
        )

        await self.db_service.AiRecalculateCityScore(city.CityID)

        return True
    
    async def _flushQuestion(
        self,
        cityID:int,
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
    
    async def _flush( self, batch: list[dict], upsert_fn,
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

    @staticmethod
    def _safe_normalized(value) -> float:
        """Return 0.0 if NormalizedValue is None or NaN, otherwise the value."""
        if value is None:
            return 0.0
        if isinstance(value, float) and math.isnan(value):
            return 0.0
        return float(value)
    
    async def _flush_pillar(
        self,
        pillar_batch: list[dict],
        source_batch: list[dict],
        *,
        force: bool = False,
    ) -> tuple[list[dict], list[dict]]:
        """Paired flush for pillar records + their source records."""
        if pillar_batch and (force or len(pillar_batch) >= _BATCH_SIZE):
            await self.db_service.bulk_upsert_pillar_evaluations(pillar_batch, source_batch)
            return [], []
        return pillar_batch, source_batch

    async def immediateSituation(self, city_id: int, **_) -> bool:
            """Score the overall city-level urban assessment."""
            year = datetime.now().year        

            ai_city= await self.db_service.get_ai_city_context(city_id, year)
            city_Name = ai_city["CityName"]
            country = ai_city["Country"]

            question = f"""
            What are the most critical recent developments, emerging risks, structural weaknesses, and key strengths across all major sectors in {city_Name}? Include insights on governance, security, economy, social cohesion, infrastructure, and institutional effectiveness. Focus on cross-pillar patterns and high-impact information relevant for executive-level city assessment and situational awareness.
            """

            document_context = await rag_query_service.get_city_document_context(city_id, question)

            if ai_city:
                ai_city_context = "\n".join(f"{key}: {value}" for key, value in ai_city.items())
            else:
                ai_city_context = ""

            ai_data = await self._ai.immediate_situation(
                        city_name=city_Name,
                        country=country,
                        ai_city_context=ai_city_context,
                        documentContext=document_context,
                        year=year
                    )

            result = self._build_immediateSituation_record(city_id, ai_data)
            
            await self.db_service.save_immediate_situation_summary(city_id,year,result)
            
            
            return True
    
    def _build_pillar_record( self, row: Any,ai_data: dict[str, Any], city_id: int,) -> dict[str, Any]:
        """Build pillar evaluation record."""

        return {
            "CityID": city_id,
            "PillarID": row.PillarID,
            "Year": self.to_int_safe(ai_data.get("Year")),
            "AIScore": self.to_float_safe(ai_data.get("AIScore")),
            "AIProgress": self.to_float_safe(ai_data.get("AIProgress")),
            "EvaluatorProgress": self.to_float_safe(row.EvaluatorProgress),
            "Discrepancy": self.to_float_safe(ai_data.get("Discrepancy")),
            "ConfidenceLevel": ai_data.get("ConfidenceLevel"),
            "EvidenceSummary": ai_data.get("EvidenceSummary"),
            "RedFlags": ai_data.get("RedFlag", ""),
            "GeographicEquityNote": ai_data.get("GeographicEquityNote"),
            "InstitutionalAssessment": ai_data.get("InstitutionalAssessment"),
            "DataGapAnalysis": ai_data.get("DataGapAnalysis"),
            "AnalystDataGapAnalysis": ai_data.get("AnalystDataGapAnalysis"),
        }


    def _build_source_records( self, row: Any, ai_data: dict[str, Any],) -> list[dict[str, Any]]:
        """Build source records for pillar evaluation."""

        sources: list[dict[str, Any]] = []

        for src in ai_data.get("Sources", []):
            sources.append({
                "CityID": row.CityID,
                "DataYear": self.to_int_safe(ai_data.get("Year")),
                "PillarID": row.PillarID,
                "SourceType": src.get("source_type"),
                "SourceName": src.get("source_name"),
                "SourceURL": src.get("source_url"),
                "DataExtract": src.get("data_extract"),
                "TrustLevel": self.to_int_safe(src.get("trust_level")),
            })

        return sources

    def _build_immediateSituation_record(self, cityId: int, ai: dict) -> dict:
            summary = ai.get("executive_summary", "")

            return {
                "CityID": cityId,
                "immediateSituationSummary": ai.get("immediateSituationSummary", "Unknown"),
                "key_developments": ai.get("key_developments", "Unknown"),
                "critical_risks": ai.get("critical_risks"),
                "gaps": ai.get("gaps"),
                "executive_summary": summary if isinstance(summary, str) and len(summary) > 50 else ""
            }

# Singleton instance
score_analyzer_service = ScoreAnalyzerService()