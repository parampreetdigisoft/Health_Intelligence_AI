"""
Data Analyzer Service - LLM-powered analysis of SQL Server data
Enhanced with Veridian Urban Index pillar-specific prompts
"""
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


class PillarPrompts:
    """Veridian Urban Index pillar-specific prompt templates"""

    PILLAR_CONTEXTS = {
    1: {
        "name": "Cleanliness and Sanitation",
        "focus": (
            "How effectively does the city manage sanitation, waste systems, drainage, "
            "public hygiene, and environmental cleanliness? Look for: solid waste collection, "
            "sewerage systems, wastewater treatment, drainage infrastructure, public hygiene "
            "campaigns, recycling programs, and sanitation access across formal and informal areas."
        ),
        "search_signals": [
            "waste collection disruption",
            "urban flooding drainage failure",
            "sanitation infrastructure upgrade",
            "public hygiene campaign",
            "sewage overflow incident",
        ],
        "red_flags": [
            "Open dumping and unmanaged waste",
            "Sewage discharge into public waterways",
            "Sanitation exclusion in informal settlements",
            "Chronic drainage collapse during rainfall",
        ],
    },

    2: {
        "name": "Smartness and Digital Readiness",
        "focus": (
            "How digitally connected and technologically capable is the city? "
            "Look for: broadband access, smart governance systems, cybersecurity readiness, "
            "digital inclusion, e-governance platforms, smart mobility systems, "
            "public Wi-Fi access, and AI-enabled urban management."
        ),
        "search_signals": [
            "smart city initiative",
            "cyberattack on city systems",
            "digital governance rollout",
            "broadband infrastructure expansion",
            "public digital services",
        ],
        "red_flags": [
            "Digital exclusion by income or geography",
            "Weak cybersecurity protections",
            "Surveillance abuse concerns",
            "Smart city branding without measurable systems",
        ],
    },

    3: {
        "name": "Conflict Risk and Early Warning",
        "focus": (
            "Are there emerging tensions, protests, violence risks, or social instability signals "
            "within the urban environment? Look for: protest activity, communal tensions, "
            "crime escalation, hate speech, forced evictions, policing disputes, "
            "and effectiveness of mediation or crisis response systems."
        ),
        "search_signals": [
            "urban protest escalation",
            "communal tension incident",
            "policing conflict",
            "civil unrest warning",
            "violent clashes city",
        ],
        "red_flags": [
            "Escalating unrest without mediation",
            "Suppressed reporting of violence",
            "Politicized policing",
            "Rapid spread of hate speech or extremist mobilization",
        ],
    },

    4: {
        "name": "Infrastructure, Mobility, and Service Delivery",
        "focus": (
            "How reliable, inclusive, and resilient are the city's infrastructure and mobility systems? "
            "Look for: electricity reliability, transport systems, public transit access, "
            "road quality, water access, ICT infrastructure, and maintenance capacity."
        ),
        "search_signals": [
            "metro or transit disruption",
            "power outage citywide",
            "water shortage urban area",
            "infrastructure investment",
            "traffic congestion crisis",
        ],
        "red_flags": [
            "Major infrastructure failures",
            "Persistent service outages",
            "Peripheral communities excluded from services",
            "Low infrastructure maintenance capacity",
        ],
    },

    5: {
        "name": "Green Infrastructure, Forests, and Urban Ecology",
        "focus": (
            "How effectively does the city protect ecological systems and integrate green infrastructure? "
            "Look for: parks, biodiversity, tree canopy, urban forests, climate adaptation measures, "
            "green corridors, and equitable access to green spaces."
        ),
        "search_signals": [
            "urban greening initiative",
            "heat mitigation project",
            "tree canopy expansion",
            "park development",
            "ecological restoration project",
        ],
        "red_flags": [
            "Unequal access to green spaces",
            "Ecological destruction from development",
            "Greenwashing without measurable impact",
            "Climate vulnerability without adaptation planning",
        ],
    },

    6: {
        "name": "Cultural Heritage, Identity, and Narrative Power",
        "focus": (
            "How does the city preserve heritage, identity, and cultural continuity? "
            "Look for: heritage conservation, monuments, museums, creative economies, "
            "minority representation, cultural funding, and protection of historic districts."
        ),
        "search_signals": [
            "heritage restoration",
            "historic district redevelopment",
            "monument controversy",
            "cultural preservation funding",
            "museum or heritage initiative",
        ],
        "red_flags": [
            "Erasure of cultural identity",
            "Redevelopment displacing historic communities",
            "Politicized heritage narratives",
            "Neglect of minority heritage sites",
        ],
    },

    7: {
        "name": "Housing and Land Security",
        "focus": (
            "Are housing systems affordable, secure, and equitable? "
            "Look for: eviction trends, housing affordability, informal settlement upgrading, "
            "land rights disputes, public housing access, and zoning inequalities."
        ),
        "search_signals": [
            "housing affordability crisis",
            "forced eviction protests",
            "public housing initiative",
            "land dispute urban area",
            "informal settlement upgrading",
        ],
        "red_flags": [
            "Mass forced evictions",
            "Speculative displacement",
            "Extreme housing inequality",
            "Lack of tenure security",
        ],
    },

    8: {
        "name": "Public Health, Inclusion, and Wellbeing",
        "focus": (
            "How accessible and inclusive are health and wellbeing systems in the city? "
            "Look for: healthcare access, mental health systems, emergency services, "
            "food security, disability inclusion, and social protection coverage."
        ),
        "search_signals": [
            "hospital capacity pressure",
            "disease outbreak urban",
            "mental health initiative",
            "healthcare access inequality",
            "public health emergency",
        ],
        "red_flags": [
            "Healthcare system overload",
            "Exclusion of vulnerable populations",
            "Unequal emergency access",
            "Public health crises without coordinated response",
        ],
    },

    9: {
        "name": "Environmental Hazards and Urban Safety",
        "focus": (
            "How exposed is the city to climate and disaster risks, and how prepared are systems "
            "to respond? Look for: flooding, heatwaves, wildfires, air quality risks, "
            "hazard mapping, emergency preparedness, and adaptation measures."
        ),
        "search_signals": [
            "urban flooding",
            "heatwave emergency",
            "air quality alert",
            "disaster preparedness activation",
            "storm impact city",
        ],
        "red_flags": [
            "Repeated unmanaged disasters",
            "Unsafe urban expansion",
            "Weak adaptation infrastructure",
            "Hazard exposure concentrated in vulnerable districts",
        ],
    },

    10: {
        "name": "Civic Resilience and Social Cohesion",
        "focus": (
            "How resilient, connected, and socially cohesive are city communities? "
            "Look for: civic participation, volunteerism, trust indicators, "
            "community resilience programs, inclusion frameworks, and social solidarity systems."
        ),
        "search_signals": [
            "community resilience initiative",
            "civic participation program",
            "social cohesion campaign",
            "neighborhood solidarity network",
            "urban inclusion initiative",
        ],
        "red_flags": [
            "Social fragmentation and polarization",
            "Declining public trust",
            "Exclusion of marginalized groups",
            "Weak civic participation",
        ],
    },

    11: {
        "name": "Business and Investment Environment",
        "focus": (
            "How attractive, fair, and stable is the city's business and investment ecosystem? "
            "Look for: business growth, startup ecosystems, licensing systems, "
            "investment inflows, SME support, commercial infrastructure, and regulatory efficiency."
        ),
        "search_signals": [
            "startup ecosystem growth",
            "foreign investment city",
            "business regulation reform",
            "commercial infrastructure project",
            "economic development initiative",
        ],
        "red_flags": [
            "Regulatory corruption",
            "Weak contract enforcement",
            "Hostile investment climate",
            "Economic exclusion of SMEs",
        ],
    },

    12: {
        "name": "Employment and Workforce Development",
        "focus": (
            "Does the city generate inclusive employment and workforce opportunities? "
            "Look for: job creation, labor force participation, skills training, "
            "youth employment, TVET systems, and workforce inclusion policies."
        ),
        "search_signals": [
            "job creation initiative",
            "youth unemployment",
            "skills training program",
            "labor market disruption",
            "workforce development strategy",
        ],
        "red_flags": [
            "Persistent unemployment",
            "Labor exploitation",
            "Skills mismatch",
            "Growth without inclusive workforce access",
        ],
    },

    13: {
        "name": "Urban Governance and Integrity",
        "focus": (
            "How transparent, accountable, and trusted are city governance systems? "
            "Look for: anti-corruption investigations, procurement transparency, "
            "audit reports, citizen participation, institutional oversight, and governance ethics."
        ),
        "search_signals": [
            "city corruption investigation",
            "procurement transparency reform",
            "municipal governance dispute",
            "audit report findings",
            "citizen participation initiative",
        ],
        "red_flags": [
            "Opaque procurement systems",
            "Institutional corruption",
            "Weak oversight mechanisms",
            "Declining trust in city leadership",
        ],
    },

    14: {
        "name": "Urban Education, Learning Ecosystems, and Knowledge Equity",
        "focus": (
            "How equitable and future-ready are urban education and learning systems? "
            "Look for: school access, learning quality, digital readiness, "
            "teacher capacity, literacy gaps, and equitable education access across districts."
        ),
        "search_signals": [
            "education reform city",
            "school infrastructure investment",
            "digital learning expansion",
            "teacher shortage",
            "education inequality",
        ],
        "red_flags": [
            "Large educational disparities",
            "School exclusion in low-income areas",
            "Digital divide in learning access",
            "Persistent dropout concentration",
        ],
    },
}
    
    @staticmethod
    def get_pillar_context(pillarId: int) -> str:
        """Get specific context and evaluation criteria for each pillar"""
        
        contexts = {
             # Urban Governance and Integrity
            13: """
                Focus: Transparency, participation, accountability, ethics, institutional capacity
                Key Evidence: Municipal budgets, procurement records, audit reports, ombudsman data,
                anti-corruption statistics, FOI response rates, council minutes
                Red Flags: Missing oversight data, zero complaints, perfect integrity claims
                Trustworthy Sources: City auditor reports, Transparency International, World Justice Project
            """,
            # Urban Education, Learning Ecosystems, and Knowledge Equity
            14: """
                Focus: Access, quality, spatial equity, digital readiness, lifelong learning
                Key Evidence: Enrollment rates, completion rates, teacher-student ratios, school mapping,
                budget allocations, inspection reports, early childhood to university coverage
                Red Flags: National-only data, dual systems (public vs private gaps), spatial inequality
                Trustworthy Sources: UNESCO Institute for Statistics, UNICEF, city education bureaus
            """,

            # Business and Investment Environment
            11: """
                Focus: Ease of doing business, property rights, dispute resolution, capital access
                Key Evidence: Business registration data, licensing portals, commercial court performance,
                land registries, investment promotion, tax structure, SME treatment
                Red Flags: Informal market contradictions, hostile regulation, weak property enforcement
                Trustworthy Sources: World Bank Enterprise Surveys, business registration agencies
            """,
            
            #Smartness and Digital Readiness
            2: """
                Focus: Digital infrastructure, e-governance, data systems, digital inclusion, cybersecurity
                Key Evidence: Broadband penetration, e-service adoption, data protection enforcement,
                cybersecurity incidents, public Wi-Fi, school connectivity, usage gaps by gender/income
                Red Flags: Smart city branding without metrics, digital inequality, vendor marketing
                Trustworthy Sources: ITU, national telecom regulators, municipal ICT offices
            """,
            
            #Cleanliness and Sanitation
            1: """
                Focus: Solid waste, liquid waste, hygiene, public cleanliness, sanitation governance
                Key Evidence: Waste collection coverage, sewerage networks, treatment plants, recycling rates,
                WASH-related disease incidence, school/market WASH audits, budget allocations
                Red Flags: CBD cleanliness vs informal settlements, missing treatment data, coverage gaps
                Trustworthy Sources: WHO/UNICEF JMP, UN-Habitat, municipal sanitation authorities
            """,
            
            #Conflict Risk and Early Warning
            3: """
                Focus: Structural drivers, protest dynamics, hate speech, early warning, mediation
                Key Evidence: Police statistics, protest/clash data, grievance logs, land disputes,
                eviction records, peace committee reports, media restrictions
                Red Flags: "No incidents" in tense environments, under-reporting, service-delivery protests
                Trustworthy Sources: ACLED, UNDP fragility diagnostics, police records
            """,
            
            #Civic Resilience and Social Cohesion
            10: """
                Focus: Trust, solidarity systems, civic participation, inclusion, community resilience
                Key Evidence: Election turnout, participatory budgeting, neighborhood associations,
                volunteer networks, trust surveys, interpersonal solidarity indicators
                Red Flags: High trust in brittle contexts, absent civil society in authoritarian settings
                Trustworthy Sources: Afrobarometer, Latinobarómetro, UNDP social cohesion assessments
            """,
            
            #Housing and Land Security
            7: """
                Focus: Tenure security, affordability, evictions, gendered land rights, spatial justice
                Key Evidence: Land registries, titling records, zoning maps, eviction data, public housing,
                informal settlement upgrading, inheritance laws, women's land rights
                Red Flags: Forced evictions, mass demolitions, gender-blind data, informal=illegitimate framing
                Trustworthy Sources: UN-Habitat, World Bank LGAF, cadastral records
            """,
            
            #Environmental Hazards and Urban Safety
            9: """
                Focus: Climate/disaster risk, hazard mapping, exposure, built environment, health risks
                Key Evidence: Hazard maps, disaster loss data, flood/heat records, air/water quality,
                building inspections, drainage plans, adaptation measures
                Red Flags: Hazard maps ignoring peripheries, no adaptation despite projections
                Trustworthy Sources: IPCC, UNDRR, EM-DAT, WHO environmental health data
            """,
            
            #Public Health, Inclusion, and Wellbeing
            8: """
                Focus: Healthcare access, mental health, disability inclusion, food security, social protection
                Key Evidence: Facility locations, staffing, service coverage, mortality data, insurance,
                emergency services, nutrition programs, disability registries, accessibility audits
                Red Flags: Averaged disparities, scarce mental health/disability data, informal settlement neglect
                Trustworthy Sources: WHO Global Health Observatory, UNICEF, health ministries
            """,
            
            #Infrastructure, Mobility, and Service Delivery
            4: """
                Focus: Water, electricity, transport, ICT, service reliability, equitable access, maintenance
                Key Evidence: Connection rates, outages, tariff structures, route maps, ridership, safety,
                maintenance budgets, road crashes, pedestrian safety, complaint systems
                Red Flags: Network presence ≠ usable access, low maintenance budgets, excluded informal transport
                Trustworthy Sources: UN-Habitat, utilities, transport authorities, World Bank
            """,
            #Green Infrastructure, Forests, and Urban Ecology
            5: """
                Focus: Urban forests, parks, biodiversity, nature-based solutions, ecological justice
                Key Evidence: Park locations/sizes, tree inventories, canopy cover, protected areas,
                biodiversity data, green corridors, climate strategies with NBS
                Red Flags: Unequal green access by income, unverified tree-planting, displacement via beautification
                Trustworthy Sources: UNEP, FAO, Global Forest Watch, parks departments
            """,
            
            #Employment and Workforce Development
            12: """
                Focus: Job creation, decent work, skills, labor rights, inclusion of marginalized workers
                Key Evidence: Labor force surveys, employment services, TVET programs, local content clauses,
                labor inspections, social security, unemployment benefits
                Red Flags: Underemployment ignored, megaprojects without skills programs, weak labor enforcement
                Trustworthy Sources: ILO, labor ministries, World Bank jobs diagnostics
            """,
            
            #Cultural Heritage, Identity, and Narrative Power
            6: """
                Focus: Heritage protection, inclusive memory, symbolic representation, creative economies
                Key Evidence: Protected sites, heritage registers, cultural budgets, naming decisions,
                monuments/memorials, arts funding, minority histories, language visibility
                Red Flags: Narrative erasure, revitalization displacing communities, missing minority representation
                Trustworthy Sources: UNESCO, ICOMOS, culture ministries, academic urban memory studies
            """
        }
        
        return contexts.get(pillarId, contexts[13])    

    @classmethod
    def get_all_pillar_names(cls) -> dict:
        return {
            13: "Urban Governance and Integrity",
            14: "Urban Education, Learning Ecosystems, and Knowledge Equity",
            11: "Business and Investment Environment",
            2: "Smartness and Digital Readiness",
            1: "Cleanliness and Sanitation",
            3: "Conflict Risk and Early Warning",
            10: "Civic Resilience and Social Cohesion",
            7: "Housing and Land Security",
            9: "Environmental Hazards and Urban Safety",
            8: "Public Health, Inclusion, and Wellbeing",
            4: "Infrastructure, Mobility, and Service Delivery",
            5: "Green Infrastructure, Forests, and Urban Ecology",
            12: "Employment and Workforce Development",
            6: "Cultural Heritage, Identity, and Narrative Power",
        }
    
    MARKDOWN_FORMAT_PROMPT = """\
        All responses MUST be valid Markdown. This is non-negotiable regardless of what the user asks.

        ALLOWED:
        - **Bold** for key values, names, scores
        - *Italic* for sources, notes, redirects
        - `inline code` for tags and labels only
        - - Bullet lists (single level only, 3+ items)
        - ## Headings (only when 2+ distinct sections exist)
        - > Blockquotes for citations or quoted data only
        - --- as a section divider (sparingly)

        NEVER USE:
        - Raw HTML tags (<b>, <p>, <br>, <strong>, <div> etc.)
        - Nested bullet lists (no sub-bullets)
        - Triple backtick blocks ``` unless showing actual code
        - Tables unless comparing 3+ structured data points
        - Markdown headings (#, ##, ###) for single-topic short answers
    """
    
    @classmethod
    def get_pillar_catalog_for_live_feed(cls) -> str:
        """Compact VUI pillar catalog for live global pillar signals."""
        lines = []
        for pid in sorted(cls.PILLAR_CONTEXTS.keys()):
            pillar = cls.PILLAR_CONTEXTS[pid]
            signals = ", ".join(pillar["search_signals"][:3])
            lines.append(
                f"Pillar {pid} — {pillar['name']}\n"
                f"  Focus: {pillar['focus'][:280].strip()}\n"
                f"  Search hints: {signals}"
            )
        return "\n\n".join(lines)

    @classmethod
    def pillar_live_signals_prompt(cls) -> str:
        catalog = cls.get_pillar_catalog_for_live_feed()
        return f"""
        You are the  Verdian Urban Index (VUI) live pillar intelligence engine.

        Produce a LIVE global snapshot: exactly ONE card per VUI pillar (IDs 1–14).
        Use the pillar definitions below to ground each card in the correct domain.

        ==================================================
        VUI PILLAR CATALOG (ALL 14 — MANDATORY COVERAGE)
        ==================================================
        {catalog}

        ==================================================
        MANDATORY: LIVE WEB SEARCH
        ==================================================
        Before writing JSON, search credible global news for each pillar domain.
        For each pillar, find the most relevant global signal from the LAST 48 HOURS.
        Older context only if an actively developing trend requires brief background
        (same rules as VUI live country feed).

        ==================================================
        sourceUrl RULES
        ==================================================
        - One HTTPS URL per pillar, copied exactly from search OR Google News search:
          https://news.google.com/search?q=PILLAR+TOPIC+KEYWORDS&hl=en-US&gl=US&ceid=US:en
        - NEVER fabricate article slugs on Reuters, BBC, AP, etc.

        ==================================================
        OUTPUT RULES
        ==================================================
        - Return EXACTLY 14 pillar objects (pillarId 1 through 14, each once).
        - title: max 55 characters — headline-style.
        - summary: max 100 characters — one clear global signal for this pillar.
        - type: "risk" or "trend" (lowercase).
        - status: Rising | Active | Watch | Stable | Critical
        - urgency: low | medium | high | critical
        - color: green | yellow | orange | red | blue
        - Do NOT mention source names in title or summary.
        - headline/subHeadline: live 48-hour framing.
        - updatedAt: current UTC ISO-8601.


        JSON format:
        {{
            "updatedAt": "2026-05-25T12:00:00Z",
            "headline": "Live Pillar Signals",
            "subHeadline": "Global peace-enabler pillar watch from the last 48 hours.",
            "pillars": [
                {{
                    "pillarId": 1,
                    "type": "risk",
                    "title": "Short headline",
                    "summary": "One sentence global signal for this pillar domain.",
                    "status": "Watch",
                    "urgency": "medium",
                    "color": "yellow",
                    "sourceUrl": "https://news.google.com/search?q=historical+memory+reconciliation&hl=en-US&gl=US&ceid=US:en"
                }}
            ]
        }}

        {_PILLAR_FEED_OUTPUT_STYLE}
        {_PILLAR_FEED_JSON_RULES}
        """