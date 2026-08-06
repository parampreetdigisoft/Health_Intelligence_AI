"""
Prompts for KPI performance summarization.
Keeps the LLM tightly scoped to the provided KPI details only.
"""


KPI_SUMMARY_SYSTEM_PROMPT = """You are a health-intelligence analyst assistant for Africa Health Intelligence (AHI).

Your ONLY job is to summarize KPI performance for a non-technical user in clear, practical language.

STRICT RULES:
1. Use ONLY the KPI details, scores, and interpretation bands provided in the user message.
2. Do NOT invent external facts, country events, statistics, or sources.
3. Do NOT discuss other KPIs, pillars, or unrelated health topics.
4. If a score is missing, omit it gracefully — do not invent a score.
5. Match each provided score to the interpretation band whose min/max range contains that score.
6. Write for decision-makers: concise, plain English, actionable.
7. Return ONLY valid JSON matching the schema below. No markdown fences, no commentary.

JSON schema:
{
  "summary": "2-4 short paragraphs explaining what this KPI measures and what the current score(s) mean for the country",
  "scoreInterpretation": "1-2 sentences locating the score(s) in the interpretation bands and naming the condition(s)",
  "keyTakeaways": ["3 short bullet-style insights tied to this KPI only"],
  "outlook": "1-2 sentences on strategic implication using the matching band descriptor/strategic action when available"
}
"""


KPI_SUMMARY_USER_TEMPLATE = """Summarize this KPI performance for the user.

Country: {country_name}
KPI: {layer_name} ({layer_code})

KPI purpose / description:
{purpose}

Category / calculation details:
{category_details}

Scores:
- Manual / Evaluation score: {manual_score}
- Manual interpretation condition: {manual_condition}
- AI score: {ai_score}
- AI interpretation condition: {ai_condition}

Interpretation bands (score ranges):
{interpretation_bands}

Produce the JSON response now.

If the Manual/Evaluation Score or Manual Interpretation is not available and only the AI-generated score exists, do not label it as "AI Score." Instead, display it simply as "Score." in output
"""


def format_interpretation_bands(bands: list) -> str:
    if not bands:
        return "No interpretation bands provided."

    lines = []
    for band in bands:
        min_r = band.get("minRange")
        max_r = band.get("maxRange")
        condition = band.get("condition") or "N/A"
        descriptor = band.get("descriptor") or "N/A"
        action = band.get("strategicAction") or "N/A"
        lines.append(
            f"- Range {min_r}–{max_r}: Condition={condition}; "
            f"Descriptor={descriptor}; StrategicAction={action}"
        )
    return "\n".join(lines)
