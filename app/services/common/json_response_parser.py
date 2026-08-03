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
        "SourceDataExtract": _with_sourcing_meta(
            analysis.get("source_data_extract"),
            analysis.get("data_quality_flag"),
            analysis.get("reporting_lag"),
        ),
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
        # Sources — lag/flag recomputed from platform Target Year
        "Sources": apply_data_sourcing_to_sources(analysis.get("sources", []), year),
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
        "key_developments": normalize_numbered_list_text(immediate.get("key_developments", "")),
        "critical_risks": normalize_numbered_list_text(immediate.get("critical_risks", "")),
        "gaps": normalize_numbered_list_text(immediate.get("gaps", "")),
        "key_findings": normalize_numbered_list_text(ai.get("key_findings", "")),
        "recommendations": normalize_numbered_list_text(ai.get("recommendations", "")),
        "executive_summary": ai.get("executive_summary", "")
    }

# ====================================================================== #
#  Internal helpers                                                      #
# ====================================================================== #

def _require_fields(data: Dict, fields: list[str]) -> None:
    for field in fields:
        if field not in data:
            raise ValueError(f"Missing required field in LLM response: '{field}'")


def normalize_numbered_list_text(value: Any) -> str:
    """
    Ensure each numbered point starts on its own line for UI/PDF readability.

    Converts legacy "||" separators and mid-line "2) / 3)" markers into newlines.
    """
    if value is None:
        return ""
    text = value if isinstance(value, str) else str(value)
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        return ""

    # Legacy pipe separator → newline
    text = re.sub(r"\s*\|\|\s*", "\n", text)
    # Numbered item mid-line (e.g. "...end. 2) Next") → newline before N)
    text = re.sub(r"\s+(?=\d+\))", "\n", text)
    # Collapse accidental blank lines between points
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


def _to_year_int(value: Any) -> Optional[int]:
    try:
        if value is None or value == "":
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None


def data_quality_flag_for_lag(lag: Optional[int], *, has_data: bool) -> str:
    if not has_data:
        return "No Data"
    if lag is None:
        return "No Data"
    if lag <= 0:
        return "Current"
    if lag == 1:
        return "1-Year Lag"
    if lag == 2:
        return "2-Year Lag"
    return "3-Year Lag"


def lag_note_for_source(data_year: int, lag: int, flag: str) -> str:
    if lag <= 0 or flag == "Current":
        return ""
    if lag == 1:
        return f"Data based on {data_year}. Current year data not available."
    if lag == 2:
        return (
            f"Data based on {data_year}. "
            "System has cascaded back 2 years to find available reporting."
        )
    return (
        f"Data sourced from {data_year}. Significant data gap of 3+ years. "
        "Consider this data a proxy estimate."
    )


_LAG_PREFIX_RE = re.compile(
    r"^(?:"
    r"\[[^\]]*Lag[^\]]*\]\s*"
    r"|\[Reporting Lag:[^\]]*\]\s*"
    r"|\d+-Year Lag note:\s*"
    r"|Data based on \d{4}\.[^.]*\.\s*"
    r"|Data sourced from \d{4}\.[^.]*\.\s*"
    r"|Significant data gap of 3\+ years\.[^.]*\.\s*"
    r"|System has cascaded back[^.]*\.\s*"
    r"|Current year data not available\.\s*"
    r"|Consider this data a proxy estimate\.\s*"
    r")+",
    re.IGNORECASE,
)


def strip_lag_prefixes(extract: str) -> str:
    text = (extract or "").strip()
    prev = None
    while prev != text:
        prev = text
        text = _LAG_PREFIX_RE.sub("", text).strip()
    return text


def apply_data_sourcing_to_sources(sources: Any, target_year: int) -> list:
    """
    Recompute reporting_lag / data_quality_flag from platform Target Year,
    and rewrite data_extract with a single correct lag note.
    """
    if not isinstance(sources, list):
        return []

    target = int(target_year)
    lookback_floor = target - 4
    normalized: list = []

    for src in sources:
        if not isinstance(src, dict):
            continue
        item = dict(src)
        data_year = _to_year_int(item.get("data_year"))
        extract = strip_lag_prefixes(str(item.get("data_extract") or ""))

        if data_year is None:
            item["reporting_lag"] = None
            item["data_quality_flag"] = "No Data"
            if extract:
                item["data_extract"] = (
                    "No data available for the last 5 years. Reporting index cannot be calculated. "
                    + extract
                ).strip()
            else:
                item["data_extract"] = (
                    "No data available for the last 5 years. Reporting index cannot be calculated."
                )
            normalized.append(item)
            continue

        lag = target - data_year
        if data_year < lookback_floor:
            flag = "3-Year Lag" if lag >= 3 else data_quality_flag_for_lag(lag, has_data=True)
        else:
            flag = data_quality_flag_for_lag(lag, has_data=True)

        item["data_year"] = data_year
        item["reporting_lag"] = max(lag, 0)
        item["data_quality_flag"] = flag

        note = lag_note_for_source(data_year, max(lag, 0), flag)
        if note:
            item["data_extract"] = f"{note} {extract}".strip()
        else:
            item["data_extract"] = extract

        normalized.append(item)

    return normalized


def _with_sourcing_meta(
    extract: Any,
    quality_flag: Any,
    reporting_lag: Any,
) -> str:
    """Prefix extract with data-quality flag / reporting lag when present."""
    text = (extract or "").strip() if isinstance(extract, str) else ("" if extract is None else str(extract))
    meta_parts: list[str] = []
    flag = (quality_flag or "").strip() if isinstance(quality_flag, str) else (
        "" if quality_flag is None else str(quality_flag).strip()
    )
    if flag:
        meta_parts.append(f"[{flag}]")
    if reporting_lag is not None and str(reporting_lag).strip() != "":
        meta_parts.append(f"[Reporting Lag: {reporting_lag}]")
    if meta_parts and not text.startswith("["):
        return f"{' '.join(meta_parts)} {text}".strip()
    return text


def _validate_ai_score(data: Dict) -> None:
    score = data.get("ai_score")
    if isinstance(score, (int, float)):
        if not (0 <= float(score) <= 100):
            raise ValueError(f"ai_score {score} is outside the valid range 0-100.")
    elif  score is not None:
        raise ValueError(
            f"ai_score must be a number 0-100, 'N/A', or 'Unknown'. Got: {score!r}"
        )


def _validate_confidence(data: Dict) -> None:
    valid = {"High", "Medium", "Low","N/A","NA", "Unknown"}
    if data.get("confidence_level") not in valid:
        logger.warning(
            "Invalid confidence_level '%s'. Defaulting to 'Medium'.",
            data.get("confidence_level"),
        )
        data["confidence_level"] = "Unknown"