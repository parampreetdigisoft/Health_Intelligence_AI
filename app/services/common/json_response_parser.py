"""
    AHI JSON Response Parser
    ------------------------
    Handles:
      - Cleaning raw LLM output into valid JSON strings
      - Fixing common JSON escaping issues
      - Validating required fields and value ranges
      - Mapping parsed dicts to the canonical DB field layout for
        question-level, pillar-level, and country-level responses
"""

import re
import json
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


# ====================================================================== #
#  Cleaning & fixing                                                      #
# ====================================================================== #

def clean_json_response(response: str) -> str:
    """
    Strip markdown fences and extract the first well-formed JSON object
    from a raw LLM response string.

    Raises:
        ValueError: if no valid JSON object can be recovered.
    """
    response = response.strip()

    # Strip ```json … ``` fences
    if response.startswith("```"):
        response = response.split("```", 2)[1]
        if response.startswith("json"):
            response = response[4:]
        response = response.strip()

    start = response.find("{")
    end = response.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("No valid JSON object found in LLM response.")

    json_str = response[start : end + 1]

    # Normalise typographic characters
    json_str = (
        json_str
        .replace("\u2018", "'")
        .replace("\u2019", "'")
        .replace("\u2013", "-")
        .replace("\u2014", "-")
        .replace("\u2026", "...")
    )

    # Strip control characters (keep \n, \r, \t for now)
    json_str = re.sub(r"[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f]", "", json_str)

    # First parse attempt
    try:
        json.loads(json_str)
        return json_str
    except json.JSONDecodeError as e:
        logger.warning(
            "Initial JSON parse failed at pos %d: %s", e.pos, e.msg
        )
        _log_context(json_str, e.pos)

    # Attempt auto-fix
    fixed = _fix_json_escaping(json_str)
    try:
        json.loads(fixed)
        logger.info("JSON successfully repaired.")
        return fixed
    except json.JSONDecodeError as e2:
        logger.error(
            "JSON repair failed at pos %d: %s\nFirst 500 chars:\n%s",
            e2.pos, e2.msg, json_str[:500],
        )
        raise ValueError(f"Could not parse JSON: {e2.msg} at position {e2.pos}")


def _fix_json_escaping(json_str: str) -> str:
    """
    Walk the string character-by-character and fix common escaping problems
    inside JSON string values:
      - Escaped single quotes (not needed in JSON)
      - Unescaped newlines / tabs inside strings
      - Invalid backslash sequences
    """
    result: list[str] = []
    i = 0
    in_string = False

    while i < len(json_str):
        char = json_str[i]

        if char == '"' and (i == 0 or json_str[i - 1] != "\\"):
            in_string = not in_string
            result.append(char)
            i += 1
            continue

        if in_string:
            if char == "\\" and i + 1 < len(json_str):
                nxt = json_str[i + 1]
                if nxt in ('"', "\\", "/", "b", "f", "n", "r", "t", "u"):
                    result.append(char)
                    result.append(nxt)
                    i += 2
                elif nxt == "'":          # escaped single quote → just the quote
                    result.append("'")
                    i += 2
                else:                     # invalid escape → double the backslash
                    result.append("\\\\")
                    i += 1
            elif char == "\n":
                result.append("\\n")
                i += 1
            elif char == "\r":
                result.append("\\r")
                i += 1
            elif char == "\t":
                result.append("\\t")
                i += 1
            else:
                result.append(char)
                i += 1
        else:
            result.append(char)
            i += 1

    return "".join(result)


def _log_context(json_str: str, pos: int, window: int = 100) -> None:
    start = max(0, pos - window)
    end = min(len(json_str), pos + window)
    logger.warning("JSON context around error: ...%s...", json_str[start:end])


# ====================================================================== #
#  Validation                                                             #
# ====================================================================== #

def validate_question_response(data: Dict) -> Dict:
    """
    Validate a parsed question-level LLM response.
    Raises ValueError on fatal problems; auto-corrects minor ones.
    """
    _require_fields(
        data,
        [
            "ai_score", "confidence_level", "evidence_summary",
            "four_layer_evidence", "temporal_scope", "distortion_screening",
            "relational_dependencies", "stress_simulation",
            "inequality_adjustment", "opacity_risk",
        ],
    )
    _validate_ai_score(data)
    _validate_confidence(data)
    return data


def validate_pillar_response(data: Dict) -> Dict:
    """Validate a parsed pillar-level LLM response."""
    _require_fields(
        data,
        ["ai_score", "confidence_level", "evidence_summary",
         "institutional_assessment", "data_gap_analysis"],
    )
    _validate_ai_score(data)
    _validate_confidence(data)
    return data


def validate_country_response(data: Dict) -> Dict:
    """Validate a parsed country-level LLM response."""
    _require_fields(
        data,
        [
            "ai_score", "confidence_level", "executive_summary",
            "cross_pillar_patterns", "institutional_capacity",
            "equity_assessment", "conflict_risk_outlook",
            "strategic_recommendation", "data_transparency_note",
            "stress_simulation", "inequality_adjustment", "opacity_risk",
        ],
    )
    _validate_ai_score(data)
    _validate_confidence(data)
    return data


# ====================================================================== #
#  Response mappers → canonical DB dicts                                  #
# ====================================================================== #

def map_question_response(
    analysis: Dict,
    pillar_id: int,
    year: int,
) -> Dict[str, Any]:
    """Map a validated question-level analysis dict to the DB field layout."""
    four = analysis.get("four_layer_evidence", {})
    stress = analysis.get("stress_simulation", {})
    return {
        "success": True,
        "CountryID": None,
        "PillarID": pillar_id,
        "Year": year,
        # Scores
        "AIScore": analysis.get("ai_score"),
        "AIProgress": analysis.get("ai_progress"),
        "ConfidenceLevel": analysis.get("confidence_level"),
        # Four-layer evidence
        "StructuralEvidence": four.get("structural"),
        "OperationalEvidence": four.get("operational"),
        "OutcomeEvidence": four.get("outcome"),
        "PerceptionEvidence": four.get("perception"),
        # Narrative fields
        "EvidenceSummary": analysis.get("evidence_summary"),
        "TemporalScope": analysis.get("temporal_scope"),
        "DistortionScreening": analysis.get("distortion_screening"),
        "RelationalDependencies": analysis.get("relational_dependencies"),
        # Stress simulation
        "StressPoliticalShock": stress.get("political_shock"),
        "StressEconomicShock": stress.get("economic_shock"),
        "StressNarrativeShock": stress.get("narrative_shock"),
        "StressOverallResilienceShock": stress.get("overall_stress_resilience"),
        # Adjustments & flags
        "InequalityAdjustment": analysis.get("inequality_adjustment"),
        "OpacityRisk": analysis.get("opacity_risk"),
        "NonCompensationNote": analysis.get("non_compensation_note"),
        "RedFlag": analysis.get("red_flag"),
        # Source fields (single primary source at question level)
        "SourceName": analysis.get("source_name"),
        "SourceType": analysis.get("source_type"),
        "SourceURL": analysis.get("source_url"),
        "SourceDataYear": analysis.get("source_data_year"),
        "SourceHierarchyLevel": analysis.get("source_trust_level"),
        "SourceDataExtract": analysis.get("source_data_extract"),
        # Optional extras
        "SourcesConsulted": analysis.get("sources_consulted"),
        "ConfidenceExplanation": analysis.get("confidence_explanation"),
    }


def map_pillar_response(
    analysis: Dict,
    pillar_id: int,
    year: int,
) -> Dict[str, Any]:
    """Map a validated pillar-level analysis dict to the DB field layout."""
    stress = analysis.get("stress_simulation", {})
    return {
        "success": True,
        "CountryID": None,
        "PillarID": pillar_id,
        "Year": year,
        # Scores
        "AIScore": analysis.get("ai_score"),
        "AIProgress": analysis.get("ai_progress"),
        "ConfidenceLevel": analysis.get("confidence_level"),
        # Narrative
        "EvidenceSummary": analysis.get("evidence_summary"),
        # Four-layer evidence
        "StructuralEvidence": analysis.get("four_layer_evidence", {}).get("structural"),
        "OperationalEvidence": analysis.get("four_layer_evidence", {}).get("operational"),
        "OutcomeEvidence": analysis.get("four_layer_evidence", {}).get("outcome"),
        "PerceptionEvidence": analysis.get("four_layer_evidence", {}).get("perception"),
        # Temporal & distortion
        "TemporalScope": analysis.get("temporal_scope"),
        "DistortionScreening": analysis.get("distortion_screening"),
        "RelationalIntegrity": analysis.get("relational_integrity"),
        # Stress simulation
        "StressPoliticalShock": stress.get("political_shock"),
        "StressEconomicShock": stress.get("economic_shock"),
        "StressNarrativeShock": stress.get("narrative_shock"),
        "StressOverallResilience": stress.get("overall_stress_resilience"),
        "StressScoreAdjustment": stress.get("stress_score_adjustment"),
        # Adjustments & flags
        "InequalityAdjustment": analysis.get("inequality_adjustment"),
        "OpacityRisk": analysis.get("opacity_risk"),
        "NonCompensationNote": analysis.get("non_compensation_note"),
        "GeographicEquityNote": analysis.get("geographic_equity_note"),
        "InstitutionalAssessment": analysis.get("institutional_assessment"),
        "DataGapAnalysis": analysis.get("data_gap_analysis"),
        "RedFlag": analysis.get("red_flag"),
        # Sources array
        "Sources": analysis.get("sources", []),
    }


def map_country_response(
    analysis: Dict,
    year: int,
) -> Dict[str, Any]:
    """Map a validated country-level analysis dict to the DB field layout."""
    four = analysis.get("four_layer_evidence", {})
    stress = analysis.get("stress_simulation", {})
    return {
        "success": True,
        "CountryID": None,
        "Year": year,
        # Scores
        "AIScore": analysis.get("ai_score"),
        "AIProgress": analysis.get("ai_progress"),
        "ConfidenceLevel": analysis.get("confidence_level"),
        "ExecutiveSummary": analysis.get("executive_summary"),
        # Four-layer evidence
        "StructuralEvidence": four.get("structural"),
        "OperationalEvidence": four.get("operational"),
        "OutcomeEvidence": four.get("outcome"),
        "PerceptionEvidence": four.get("perception"),
        # Temporal & distortion
        "TemporalScope": analysis.get("temporal_scope"),
        "DistortionScreening": analysis.get("distortion_screening"),
        # Stress simulation
        "PoliticalShock": stress.get("political_shock"),
        "EconomicShock": stress.get("economic_shock"),
        "NarrativeShock": stress.get("narrative_shock"),
        "OverallStressResilience": stress.get("overall_stress_resilience"),
        "StressScoreAdjustment": stress.get("stress_score_adjustment"),
        # Adjustments, patterns & flags
        "InequalityAdjustment": analysis.get("inequality_adjustment"),
        "OpacityRisk": analysis.get("opacity_risk"),
        "NonCompensationNote": analysis.get("non_compensation_note"),
        "CrossPillarPatterns": analysis.get("cross_pillar_patterns"),
        "RelationalIntegrity": analysis.get("relational_integrity"),
        "InstitutionalCapacity": analysis.get("institutional_capacity"),
        "EquityAssessment": analysis.get("equity_assessment"),
        "ConflictRiskOutlook": analysis.get("conflict_risk_outlook"),
        "StrategicRecommendation": analysis.get("strategic_recommendation"),
        "DataTransparencyNote": analysis.get("data_transparency_note"),
        "PrimarySource": analysis.get("primary_source"),
    }
def build_immediateSituation_record(ai: dict) -> Dict[str, Any]:
    immediate = ai.get("immediateSituation", {}) or {}

    return {
        "immediateSituationSummary": immediate.get("summary", ""),
        "key_developments": immediate.get("key_developments", ""),
        "critical_risks": immediate.get("critical_risks", ""),
        "gaps": immediate.get("gaps", ""),
        "executive_summary": ai.get("executive_summary", "")
    }

# ====================================================================== #
#  Internal helpers                                                      #
# ====================================================================== #

def _require_fields(data: Dict, fields: list[str]) -> None:
    for field in fields:
        if field not in data:
            raise ValueError(f"Missing required field in LLM response: '{field}'")


def _validate_ai_score(data: Dict) -> None:
    score = data.get("ai_score")
    if isinstance(score, (int, float)):
        if not (0 <= float(score) <= 100):
            raise ValueError(f"ai_score {score} is outside the valid range 0-4.")
    elif score not in ("N/A", "Unknown",None):
        raise ValueError(
            f"ai_score must be a number 0-100, 'N/A', or 'Unknown'. Got: {score!r}"
        )


def _validate_confidence(data: Dict) -> None:
    valid = {"High", "Medium", "Low"}
    if data.get("confidence_level") not in valid:
        logger.warning(
            "Invalid confidence_level '%s'. Defaulting to 'Medium'.",
            data.get("confidence_level"),
        )
        data["confidence_level"] = "Medium"