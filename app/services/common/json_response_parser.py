"""
    VUI JSON Response Parser
    ------------------------
    Handles:
      - Cleaning raw LLM output into valid JSON strings
      - Fixing common JSON escaping issues
      - Validating required fields and value ranges
      - Mapping parsed dicts to the canonical DB field layout for
        question-level, pillar-level, and city-level responses
"""

import datetime
import re
import json
import logging
from typing import Any, Dict, Optional
from datetime import datetime
from typing import Dict, Any
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

# def validate_question_response(data: Dict) -> Dict:
#     """
#     Validate a parsed question-level LLM response.
#     Raises ValueError on fatal problems; auto-corrects minor ones.
#     """

#     _require_fields(
#         data,
#         [
#             "ai_score",
#             "ai_progress",
#             "confidence_level",
#             "evidence_summary",
#             "four_layer_evidence",
#             "temporal_scope",
#             "distortion_screening",
#             "relational_dependencies",
#             "stress_simulation",
#             "inequality_adjustment",
#             "opacity_risk",
#             "sources",  # switched to structured sources (better design)
#         ],
#     )

#     _validate_ai_score(data)
#     _validate_ai_progress(data)
#     _validate_confidence(data)
#     _validate_sources(data)

#     return data


def validate_question_response(data: Dict) -> Dict:
    """
    Validate a parsed question-level LLM response.
    Raises ValueError on fatal problems; auto-corrects minor ones.
    """

    _require_fields(
        data,
        [
            "ai_score",
            "ai_progress",
            "confidence_level",
            "evidence_summary",
            "data_sources_count",
            "source_type",
            "source_name",
            "source_url",
            "source_data_year",
            "source_trust_level",
            "source_data_extract",
        ],
    )

    # Optional fields
    data.setdefault("red_flag", "")
    data.setdefault("geographic_equity_note", "")

    _validate_ai_score(data)
    _validate_ai_progress(data)
    _validate_confidence(data)
    _validate_source(data)

    return data

def _validate_ai_score(data: Dict) -> None:
    score = data.get("ai_score")

    if isinstance(score, (int, float)):
        if not (0 <= float(score) <= 4):
            raise ValueError(f"ai_score {score} is outside the valid range 0-4.")
    elif score not in ("N/A", "Unknown"):
        raise ValueError(
            f"ai_score must be a number 0-4, 'N/A', or 'Unknown'. Got: {score!r}"
        )


def _validate_ai_progress(data: Dict) -> None:
    progress = data.get("ai_progress")

    if not isinstance(progress, (int, float)) or not (0 <= progress <= 100):
        raise ValueError(f"ai_progress must be 0-100, got {progress}")
    
def _validate_sources(data: Dict) -> None:
    sources = data.get("sources")

    if not isinstance(sources, list) or len(sources) == 0:
        logger.warning("Invalid sources array, creating placeholder")

        data["sources"] = [{
            "source_type": "Unknown",
            "source_name": "Data not available",
            "source_url": "Not available",
            "data_year": datetime.now().year,
            "trust_level": 1,
            "data_extract": "Insufficient data available"
        }]
        return

    for src in sources:
        _validate_single_source(src)

def _validate_single_source(src: Dict) -> None:
    required = [
        "source_type",
        "source_name",
        "source_url",
        "data_year",
        "trust_level",
        "data_extract",
    ]

    for field in required:
        if field not in src:
            raise ValueError(f"Missing source field: {field}")

    # Trust level check
    if not (1 <= src["trust_level"] <= 7):
        raise ValueError(
            f"source_trust_level must be 1-7, got {src['trust_level']}"
        )
        
def validate_pillar_response(data: Dict) -> Dict:
    """Validate a parsed pillar-level LLM response."""

    _require_fields(
        data,
        [
            "ai_score",
            "ai_progress",
            "confidence_level",
            "evidence_summary",
            "institutional_assessment",
            "data_gap_analysis",
            "sources",  # restored
        ],
    )

    _validate_ai_score(data)
    _validate_ai_progress(data)
    _validate_confidence(data)
    _validate_sources(data)

    return data


def validate_city_response(data: Dict) -> Dict:
    """Validate a parsed city-level LLM response."""

    _require_fields(
        data,
        [
            "ai_score",
            "ai_progress",
            "confidence_level",
            "city_profile",
            "peer_comparison",
            "evidence_summary",
            "source",
            "cross_pillar_patterns",
            "institutional_capacity",
            "equity_assessment",
            "sustainability_outlook",
        ],
    )

    _validate_ai_score(data)
    _validate_ai_progress(data)
    _validate_confidence(data)

    return data

def _validate_source(data: Dict) -> None:
    required = [
        "source_type",
        "source_name",
        "source_url",
        "source_data_year",
        "source_trust_level",
        "source_data_extract",
    ]

    for field in required:
        if field not in data:
            raise ValueError(f"Missing source field: {field}")

    trust_level = data.get("source_trust_level")

    if not isinstance(trust_level, (int, float)) or not (1 <= trust_level <= 7):
        raise ValueError(
            f"source_trust_level must be 1-7, got {trust_level}"
        )

    source_count = data.get("data_sources_count")

    if not isinstance(source_count, int) or source_count < 0:
        raise ValueError(
            f"data_sources_count must be a non-negative integer, got {source_count}"
        )
# ====================================================================== #
#  Response mappers → canonical DB dicts                                  #
# ====================================================================== #

def map_question_response(
    analysis: Dict,
    pillar_id: int,
    year: int,
) -> Dict[str, Any]:
    """Map a validated question-level analysis dict to the DB field layout."""

    return {
        "success": True,
        "CityID": None,
        "PillarID": pillar_id,
        "Year": year,

        # Scores
        "AIScore": analysis.get("ai_score"),
        "AIProgress": analysis.get("ai_progress"),
        "ConfidenceLevel": analysis.get("confidence_level"),
        "Discrepancy": analysis.get("discrepancy"),

        # Core narrative
        "EvidenceSummary": analysis.get("evidence_summary"),
        "RedFlag": analysis.get("red_flag", ""),
        "GeographicEquityNote": analysis.get("geographic_equity_note", ""),

        # Source metadata
        "DataSourcesCount": analysis.get("data_sources_count"),
        "SourceType": analysis.get("source_type"),
        "SourceName": analysis.get("source_name"),
        "SourceURL": analysis.get("source_url"),
        "SourceDataYear": analysis.get("source_data_year"),
        "SourceHierarchyLevel": analysis.get("source_trust_level"),
        "SourceDataExtract": analysis.get("source_data_extract"),
    }




def map_pillar_response(
    analysis: Dict,
    pillar_id: int,
    pillar_name: str,
    year: int,
    discrepancy: float = None
) -> Dict[str, Any]:
    """Map a validated pillar-level analysis dict to the DB field layout."""

    return {
        "success": True,
        "CityID": None,
        "PillarID": pillar_id,
        "PillarName": pillar_name,
        "Year": year,

        # Scores
        "AIScore": analysis.get("ai_score"),
        "AIProgress": analysis.get("ai_progress"),
        "Discrepancy": discrepancy,
        "ConfidenceLevel": analysis.get("confidence_level"),

        # Core narrative
        "EvidenceSummary": analysis.get("evidence_summary"),
        "RedFlag": analysis.get("red_flag", ""),
        "GeographicEquityNote": analysis.get("geographic_equity_note", ""),

        # Institutional analysis
        "InstitutionalAssessment": analysis.get("institutional_assessment", ""),
        "DataGapAnalysis": analysis.get("data_gap_analysis", ""),
        "AnalystDataGapAnalysis": analysis.get("analyst_data_gap_analysis", ""),

        # Sources (array)
        "Sources": analysis.get("sources", []),

        # Metadata
        "Timestamp": datetime.now().isoformat()
    }


def map_city_response(
    analysis: Dict,
    city_name: str,
    year: int,
    discrepancy: float = None
) -> Dict[str, Any]:
    """Map a validated country/city-level analysis dict to the DB field layout."""

    return {
        "success": True,
        "CityID": None,
        "CityName": city_name,
        "Year": year,

        # Scores
        "AIScore": analysis.get("ai_score"),
        "AIProgress": analysis.get("ai_progress"),
        "Discrepancy": discrepancy,
        "ConfidenceLevel": analysis.get("confidence_level"),

        # Combined summary (city_profile + evidence_summary)
        "EvidenceSummary": (
            (analysis.get("city_profile") or "") +
            ("\n\n" if analysis.get("city_profile") else "") +
            (analysis.get("evidence_summary") or "")
        ),
        # Source (single field in your return)
        "Source": analysis.get("source"),

        # Strategic / analytical fields
        "CrossPillarPatterns": analysis.get("cross_pillar_patterns", ""),
        "InstitutionalCapacity": analysis.get("institutional_capacity", ""),
        "EquityAssessment": analysis.get("equity_assessment", ""),
        "SustainabilityOutlook": analysis.get("sustainability_outlook", ""),
        "StrategicRecommendation": analysis.get("strategic_recommendation", ""),
        "DataTransparencyNote": analysis.get("data_transparency_note", ""),
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
        if not (0 <= float(score) <= 4):
            raise ValueError(f"ai_score {score} is outside the valid range 0-4.")
    elif score not in ("N/A", "Unknown"):
        raise ValueError(
            f"ai_score must be a number 0-4, 'N/A', or 'Unknown'. Got: {score!r}"
        )


def _validate_confidence(data: Dict) -> None:
    valid = {"High", "Medium", "Low"}
    if data.get("confidence_level") not in valid:
        logger.warning(
            "Invalid confidence_level '%s'. Defaulting to 'Medium'.",
            data.get("confidence_level"),
        )
        data["confidence_level"] = "Medium"

def _validate_confidence(data: Dict) -> None:
    valid = {"High", "Medium", "Low"}
    if data.get("confidence_level") not in valid:
        logger.warning(
            "Invalid confidence_level '%s'. Defaulting to 'Medium'.",
            data.get("confidence_level"),
        )
        data["confidence_level"] = "Medium"

@staticmethod
def _calculate_discrepancy(        
        ai_progress: float, 
        evaluator_score: Optional[float]
    ) -> float:
        """Calculate discrepancy between AI and evaluator scores"""
        if evaluator_score is not None:

            return abs(ai_progress - evaluator_score)
        return ai_progress