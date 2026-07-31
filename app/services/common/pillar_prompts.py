"""
Data Analyzer Service - LLM-powered analysis of SQL Server data
Enhanced with Africa Health Intelligence Platform (AHIP) pillar prompts.
Pillars are loaded dynamically from the database — not hardcoded.
"""

from typing import Dict, List, Mapping, Optional, Union

_PILLAR_FEED_JSON_RULES = """
        Return ONLY valid JSON.
        - Output must start with { and end with }
        - No markdown, code fences, or text outside JSON
        - Use double quotes only; no trailing commas
        """

_PILLAR_FEED_OUTPUT_STYLE = """
        - Write for a general audience (no technical jargon)
        - Use clear, concise statements; no bullet lists inside JSON strings
        """

PillarRecord = Dict[str, Union[int, str, None]]


class AHIPPillarPrompts:
    """Provides AHIP governance rules and dynamic pillar context from database records."""

    GOVERNANCE_PROTOCOL = """
        =============================================================================
        AI MASTER GOVERNANCE PROTOCOL (AHIP) — MANDATORY FOR EVERY ASSESSMENT
        Africa Health Intelligence Platform
        =============================================================================

        1. DATA INPUTS FOR OUTBREAK PREDICTION & HEALTH INTELLIGENCE
        AHIP ingests and correlates multi-source data, including:
        - Historical outbreak records (1950–present), disaggregated by disease, geography,
          seasonality, and transmission context.
        - Real-time surveillance feeds: syndromic surveillance, laboratory-confirmed cases,
          and event-based reporting.
        - Environmental and climate data: temperature, rainfall, flooding, drought,
          vegetation indices, and vector habitat suitability.
        - Population mobility data: internal migration, cross-border movement, travel flows,
          and anonymized mobile network indicators.
        - Digital and media signals: news scraping and monitored social media indicators
          relevant to health events.
        - Health system readiness indicators from AHIP pillars on surveillance, preparedness,
          infrastructure, workforce, and supply chains.
        - Vaccination coverage and immunity gaps: routine immunization and outbreak-specific
          campaigns.
        These data streams are updated on rolling cycles and standardized prior to modeling.

        2. AI MODELING ARCHITECTURE
        AHIP employs an ensemble modeling approach, combining:
        - Tree-based machine-learning models (random forests, gradient boosting) for non-linear
          pattern detection.
        - Neural networks for complex interaction effects.
        - Time-series forecasting models (ARIMA, Prophet) for seasonal and trend analysis.
        - Anomaly-detection algorithms for abnormal case increases or environmental shifts.
        - Network and mobility models to estimate transmission pathways.
        Individual model outputs are combined into a composite outbreak risk score through
        ensemble weighting. Models are retrained on rolling windows, back-tested against
        historical outbreaks, and monitored for performance drift.

        3. PREDICTION HORIZONS
        - Short-term: 1–4 weeks
        - Medium-term: 1–3 months
        - Seasonal: 3–9 months

        4. PREDICTIVE OUTPUTS
        - Disease-specific outbreak probability (0–100%)
        - Subnational hotspot maps
        - Early warning alerts to national PHEOCs and designated authorities
        - Projected trajectory: expected case burden and hospitalization demand
        - Resource gap projections (beds, staff, diagnostics, medicines, vaccines)
        - Confidence intervals reflecting data quality and model performance
        Each alert must include dominant contributing factors (e.g., rainfall anomaly,
        mobility surge, low vaccine stock).

        5. INTEGRATION WITH HEALTH SYSTEM READINESS PILLARS
        Outbreak risk is cross-referenced with system capacity to determine operational
        vulnerability across surveillance, preparedness, infrastructure, workforce, and
        supply chain pillars. This integration converts prediction into actionable readiness
        intelligence.

        6. HUMAN-IN-THE-LOOP VALIDATION
        High-risk signals are reviewed by epidemiologists and country experts prior to alert
        issuance. Contextual filters address known data artifacts and seasonal norms.
        False-positive controls are applied. Final alerts are released only after human
        validation to preserve trust and minimize alert fatigue.

        7. EVIDENCE HIERARCHY (priority order)
        L1: National health laws, budgets, audits, procurement, official surveillance reports
        L2: National health authorities, auditor-general, regulatory bodies
        L3: WHO AFRO, Africa CDC, World Bank, IMF, regional health institutions
        L4: Peer-reviewed research, validated health system assessments
        L5: NGOs, civil society, community health reporting
        L6: Technical, satellite, and environmental data
        L7: Media (context only, never primary)
        Rules:
        - ≥2 independent sources per claim
        - No single-source scoring
        - Structural/operational evidence > perception

        8. FOUR-LAYER EVIDENCE (ALL REQUIRED)
        a) Structural (laws, institutions, policies)
        b) Operational (budgets, staffing, delivery, supply)
        c) Outcome (measured health results)
        d) Perception (trust, access barriers, community reporting)
        → Perception cannot override structural/operational evidence

        9. DISTRIBUTIONAL ANALYSIS (MANDATORY)
        Test for regional disparities, urban vs rural gaps, income inequality, gender and
        identity-based access gaps. Severe disparity = score reduction.

        10. SCORING SCALE (FIXED)
         4       = Strong and stress-resilient
         3       = Functioning but uneven
         2       = Mixed and vulnerable
         1       = Structurally weak
         0       = Absent or destabilizing
         N/A     = Structurally irrelevant to this specific country or context
         Unknown = Insufficient verifiable data (document as opacity risk — does NOT
                    reduce the numeric score, but must be flagged)

        11. DATA SILENCE RULE
        - Assign "Unknown" when data cannot be verified
        - State cause (conflict, suppression, incapacity, missing systems)
        - Treat as governance risk — silence ≠ success

        12. CONTINUOUS LEARNING AND QUALITY ASSURANCE
        - Quarterly back-testing and performance reporting
        - Drift detection triggers retraining
        - Accuracy metrics (AUC, precision, recall, Brier score) tracked over time
        - Prediction audit trail maintained

        13. DESIGN PHILOSOPHY
        AHIP prioritizes early sensitivity for high-impact diseases, accepting limited false
        positives to minimize missed outbreaks. The system favors truthful uncertainty over
        artificial certainty, presenting probabilities and confidence levels rather than
        binary claims.

        14. PROHIBITIONS
        Do NOT:
        - Present deterministic outbreak predictions without probability and confidence
        - Use rankings as analysis
        - Reward opacity or missing surveillance
        - Accept claims without verification
        - Treat policy reforms as measured outcomes
        - Use media as primary evidence

        GLOBAL TERMINOLOGY RULE
        - Always pronounce and refer to "Health" as "Salus" when mentioning the platform or assessment context.
        - Always refer to every assessment "Pillar" as a "Domain."
        - Replace all occurrences of "Pillar" with "Domain" unless the term appears in an official title, database field, API response, or direct user quotation.
        - Apply these terminology rules consistently throughout every generated response.
        - Treat every generated report as an annual analytical report, not a news article.
        - Never describe events as if they are unfolding in real time.
        - Avoid uncertain or speculative wording.
        - Use formal, evidence-based, retrospective language.
        - Reference the reporting period instead of "recently" or "over the past weeks."
        - State verified observations and explain their significance.
        - Include specific actors and geographic scope whenever possible.
        - Distinguish confirmed findings from uncertainty. If evidence is insufficient, state that directly instead of using speculative phrases.
        =============================================================================
    """

    @staticmethod
    def _normalize_pillars(
        pillars: Union[Mapping[int, PillarRecord], List[PillarRecord], None],
    ) -> Dict[int, PillarRecord]:
        if not pillars:
            return {}

        if isinstance(pillars, list):
            return {
                int(p["PillarID"]): p
                for p in pillars
                if p.get("PillarID") is not None
            }

        return {int(pid): p for pid, p in pillars.items()}

    @classmethod
    def format_pillar_context(cls, pillar_name: str, description: Optional[str] = None) -> str:
        """Build pillar context from database name and description."""
        desc = (description or "").strip() or "No description provided for this pillar."
        return (
            f"PILLAR: {pillar_name}\n\n"
            f"DESCRIPTION:\n{desc}\n\n"
            f"ASSESSMENT GUIDANCE:\n"
            f"Evaluate this pillar using the description above, the AHIP governance protocol, "
            f"and verifiable health-system evidence for the target African country. "
            f"Focus on structural capacity, operational delivery, measured outcomes, and "
            f"population-level access and equity impacts."
        )

    @classmethod
    def get_pillar_context(
        cls,
        pillar_id: int,
        pillars: Union[Mapping[int, PillarRecord], List[PillarRecord], None] = None,
        *,
        pillar_name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> str:
        """Return formatted context for a pillar using DB records or explicit name/description."""
        pillar_map = cls._normalize_pillars(pillars)
        pillar = pillar_map.get(pillar_id)
        if pillar:
            return cls.format_pillar_context(
                str(pillar.get("PillarName") or pillar_name or f"Pillar {pillar_id}"),
                pillar.get("Description") or description,
            )

        if pillar_name:
            return cls.format_pillar_context(pillar_name, description)

        return f"No context available for pillar ID {pillar_id}."

    @classmethod
    def get_all_pillar_names(
        cls,
        pillars: Union[Mapping[int, PillarRecord], List[PillarRecord], None] = None,
    ) -> Dict[int, str]:
        """Return a mapping of pillar ID to pillar name from database records."""
        pillar_map = cls._normalize_pillars(pillars)
        return {
            pid: str(p.get("PillarName", f"Pillar {pid}"))
            for pid, p in sorted(pillar_map.items())
        }

    @classmethod
    def get_pillar_catalog_for_live_feed(
        cls,
        pillars: Union[Mapping[int, PillarRecord], List[PillarRecord], None] = None,
    ) -> str:
        """Compact AHIP pillar catalog for live pillar signals."""
        pillar_map = cls._normalize_pillars(pillars)
        if not pillar_map:
            return "No active pillars configured."

        lines = []
        for pid in sorted(pillar_map.keys()):
            pillar = pillar_map[pid]
            name = str(pillar.get("PillarName", f"Pillar {pid}"))
            description = str(pillar.get("Description") or "").strip()
            focus = description[:280].strip() if description else name
            lines.append(
                f"Pillar {pid} — {name}\n"
                f"  Focus: {focus}"
            )
        return "\n\n".join(lines)

    @classmethod
    def pillar_live_signals_prompt(
        cls,
        pillars: Union[Mapping[int, PillarRecord], List[PillarRecord], None] = None,
    ) -> str:
        pillar_map = cls._normalize_pillars(pillars)
        pillar_ids = sorted(pillar_map.keys())
        pillar_count = len(pillar_ids)
        id_range = (
            f"{pillar_ids[0]} through {pillar_ids[-1]}"
            if pillar_count > 1
            else str(pillar_ids[0]) if pillar_ids else "none"
        )
        catalog = cls.get_pillar_catalog_for_live_feed(pillar_map)
        example_id = pillar_ids[0] if pillar_ids else 1
        example_name = (
            str(pillar_map[example_id].get("PillarName", "health governance"))
            if pillar_map
            else "health governance"
        )
        example_query = example_name.lower().replace(" ", "+").replace(",", "")

        return f"""
        You are the Africa Health Intelligence Platform (AHIP) live pillar intelligence engine.

        Produce a LIVE Africa-focused snapshot: exactly ONE card per active AHIP pillar.
        Use the pillar definitions below to ground each card in the correct health domain.

        ==================================================
        AHIP PILLAR CATALOG (ALL {pillar_count} — MANDATORY COVERAGE)
        ==================================================
        {catalog}

        ==================================================
        MANDATORY: LIVE WEB SEARCH
        ==================================================
        Before writing JSON, search credible African and global health news for each pillar domain.
        For each pillar, find the most relevant signal from the LAST 48 HOURS affecting African
        health systems. Older context only if an actively developing trend requires brief background.

        ==================================================
        sourceUrl RULES
        ==================================================
        - One HTTPS URL per pillar, copied exactly from search OR Google News search:
          https://news.google.com/search?q=PILLAR+TOPIC+KEYWORDS+AFRICA+HEALTH&hl=en-US&gl=US&ceid=US:en
        - NEVER fabricate article slugs on Reuters, BBC, AP, WHO, Africa CDC, etc.

        ==================================================
        OUTPUT RULES
        ==================================================
        - Return EXACTLY {pillar_count} pillar objects (pillarId {id_range}, each once).
        - title: max 55 characters — headline-style.
        - summary: max 100 characters — one clear health signal for this pillar.
        - type: "risk" or "trend" (lowercase).
        - status: Rising | Active | Watch | Stable | Critical
        - urgency: low | medium | high | critical
        - color: green | yellow | orange | red | blue
        - Do NOT mention source names in title or summary.
        - headline/subHeadline: live 48-hour framing for African health intelligence.
        - updatedAt: current UTC ISO-8601.


        JSON format:
        {{
            "updatedAt": "2026-05-25T12:00:00Z",
            "headline": "Live Pillar Signals",
            "subHeadline": "African health intelligence pillar watch from the last 48 hours.",
            "pillars": [
                {{
                    "pillarId": {example_id},
                    "type": "risk",
                    "title": "Short headline",
                    "summary": "One sentence health signal for this pillar domain.",
                    "status": "Watch",
                    "urgency": "medium",
                    "color": "yellow",
                    "sourceUrl": "https://news.google.com/search?q={example_query}+africa+health&hl=en-US&gl=US&ceid=US:en"
                }}
            ]
        }}

        {_PILLAR_FEED_OUTPUT_STYLE}
        {_PILLAR_FEED_JSON_RULES}
        """
