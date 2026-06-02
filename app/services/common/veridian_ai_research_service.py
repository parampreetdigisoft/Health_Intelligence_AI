"""
    Veridian Urban Index AI Research Service
    Independent research-based scoring with evidence tracking
"""
import re
import json
import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from app.config import settings
from app.services.common.city_prompt import VerdianPromptTemplates
from app.services.common.llm_base_service import LLMBaseService
from app.services.common.llm_factory import llm_factory
from app.services.common.pillar_prompts import PillarPrompts
from app.services.common import json_response_parser as jrp

logger = logging.getLogger(__name__)

_CITY_USER_TMPL = """
    City: {city_name}
    Country: {country}
    Year: {year}
"""
class VerdianAIResearchService:
    """AI service that conducts independent research and evidence-based scoring"""

    def __init__(self):
        self.llm = None
        self._initialized = False
        self.max_retries = 3
        self.retry_delay = 1  # seconds
        self._llm_svc = LLMBaseService(max_retries=3, retry_delay=1.0)

    async def initialize(self):
        """Initialize the LLM with retry logic"""
        if self._initialized:
            return

        for attempt in range(self.max_retries):
            try:
                self.llm = llm_factory.create_llm()
                self._initialized = True
                logger.info(f"Veridian AI Research Service initialized with {settings.LLM_PROVIDER}")
                return
            except Exception as e:
                logger.error(f"Initialization attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay)
                    continue
                else:
                    raise RuntimeError(f"Failed to initialize after {self.max_retries} attempts: {e}")

    async def _ensure_initialized(self):
        """Ensure LLM is initialized before use"""
        if not self._initialized or self.llm is None:
            await self.initialize()

    async def research_and_score_question(
            self,
            city_name: str,
            city_address: str,
            pillarID:int,
            pillar_name: str,
            question_text: str,
            scoreProgress: Optional[float] = None,
            evaluator_score: Optional[float] = None,
            year: int = None
        ) -> Dict[str, Any]:
            """
            Conduct independent research for a single question with enhanced validation
            
            Returns comprehensive evidence-based scoring with detailed source tracking
            """
            try:
                await self._ensure_initialized()
                
                if year is None:
                    year = datetime.now().year
                
                pillar_context = PillarPrompts.get_pillar_context(pillarID)

                prompt = ChatPromptTemplate.from_messages([
                    ("system", VerdianPromptTemplates._get_question_system_prompt(self, city_name, city_address, scoreProgress, evaluator_score, pillar_context)),
                    ("user", """Conduct independent research and provide evidence-based scoring.
                     
                    City: {city_name}
                    Address: {city_address}
                    Question: {question_text}
                    Pillar: {pillar_name}
                    Year: {year}
                    {evaluator_context}

                    SEARCH THE WEB for verifiable evidence and provide your assessment.
                    
                    Remember: Return ONLY a single JSON object with the EXACT structure specified. Report details for only the MOST TRUSTWORTHY source.""")
                ])

                evaluator_context = ""
                if evaluator_score is not None:
                    evaluator_context = f"Evaluator Score: {evaluator_score}/4, Progress: {scoreProgress}%" if evaluator_score else "No evaluator score provided"

                     # Execute with retry logic
                for attempt in range(self.max_retries):
                    try:

                        chain = prompt | self.llm | StrOutputParser()
                        
                        result = await chain.ainvoke({
                            "city_name": city_name,
                            "city_address": city_address,
                            "question_text": question_text,
                            "pillar_name": pillar_name,
                            "pillar_context": pillar_context,
                            "year": year,
                            "evaluator_score": evaluator_score if evaluator_score is not None else "Not provided",
                            "scoreProgress": scoreProgress if scoreProgress is not None else 0,
                            "evaluator_context": evaluator_context
                        })
                        
                        if not result or result.strip() == "{}":
                            continue  # retry

                        # Parse and validate response
                        analysis = json.loads(jrp.clean_json_response(result))
                        jrp.validate_question_response(analysis)                          
                        
                        mapped = jrp.map_question_response(analysis, pillarID, year)
                        if evaluator_score is not None and analysis.get("ai_progress") is not None:
                            mapped["Discrepancy"] = jrp._calculate_discrepancy(
                                analysis.get("ai_progress"),
                                evaluator_score
                            )
                        else:
                            mapped["Discrepancy"] = None

                        return mapped


                    except (json.JSONDecodeError, ValueError) as e:
                        logger.error(f"JSON parse error on attempt {attempt + 1}: {e}")
                        if attempt < self.max_retries - 1:
                            await asyncio.sleep(self.retry_delay)
                            continue
                        else:
                            raise

            except Exception as e:
                logger.error(f"Error in question research: {e}", exc_info=True)
                return {
                    "success": False,
                    "error": str(e)
                }
            
    async def research_and_score_pillar(
        self,
        city_name: str,
        city_address: str,
        pillarId: int,
        pillar_name: str,
        questions_context: str = None,
        evaluator_score: Optional[float] = None,
        aIScore: Optional[float] = None,
        year: int = None
    ) -> Dict[str, Any]:
        """
        Conduct independent research for an entire pillar with comprehensive web search.
        
        Args:
            city_name: Name of the city
            city_address: Full address of the city
            pillarId: ID of the pillar (1-14)
            pillar_name: Name of the pillar
            questions_context: Context from pillar questions with scores
            evaluator_score: Human evaluator's pillar score (0-4)
            aIScore: Previous AI score for this pillar (0-4)
            year: Assessment year
        
        Returns:
            Dictionary with research results and scoring
        """
        try:
            await self._ensure_initialized()
            
            if year is None:
                year = datetime.now().year
            
            # Get pillar-specific context
            pillar_context = PillarPrompts.get_pillar_context(pillarId)
            
            # Build human context
            evaluator_context = (
                f"Human Evaluator Score: {evaluator_score}/100"
                if evaluator_score is not None
                else "No evaluator score"
            )
            
            # Build AI context
            ai_input_context = (
                f"Previous AI Score: {aIScore}/4"
                if aIScore is not None
                else "No previous AI score available."
            )
            
            # Create the prompt
            prompt = ChatPromptTemplate.from_messages([
                ("system",VerdianPromptTemplates._get_pillar_system_prompt(self, city_name, pillar_name, year, evaluator_context, ai_input_context, pillar_context)),
                (
                    "user",
                    """Research and score the following pillar:

                    City: {city_name}
                    Full Address: {city_address}
                    Pillar: {pillar_name}
                    Assessment Year: {year}

                    Conduct comprehensive web research using the search strategies outlined above. Find real evidence from trustworthy sources and provide your independent scoring with clear justification.

                    Remember: Search for official data, government reports, international organization data, and academic research. Provide verifiable evidence-based scoring."""
                )
            ])
            
                       # Execute with retry logic
            for attempt in range(self.max_retries):
                try:
                    chain = prompt | self.llm | StrOutputParser()
                    
                    result = await chain.ainvoke({
                        "city_name": city_name,
                        "city_address": city_address,
                        "pillar_name": pillar_name,
                        "year": year,
                        "pillar_context": pillar_context,
                        "ai_input_context": ai_input_context,
                        "evaluator_context": evaluator_context
                    })
                    
                    if not result or result.strip() == "{}":
                        continue  # retry
                
                    # Parse and validate                   

                    analysis = json.loads(jrp.clean_json_response(result))
                    jrp.validate_pillar_response(analysis)                    
                   
                    discrepancy = None
                    if evaluator_score is not None and analysis.get("ai_progress") is not None:
                        discrepancy = jrp._calculate_discrepancy(
                            analysis.get("ai_progress"),
                            evaluator_score
                        )                   
                    mapped = jrp.map_pillar_response(
                        analysis=analysis,
                        pillar_id=pillarId,
                        pillar_name=pillar_name,
                        year=year,
                        discrepancy=discrepancy
                    )

                    return mapped
                        
                except (json.JSONDecodeError, ValueError) as e:
                    logger.error(f"JSON parse error on attempt {attempt + 1}: {e}")
                    if attempt < self.max_retries - 1:
                        await asyncio.sleep(self.retry_delay)
                        continue
                    else:
                        raise
        except Exception as e:
            logger.error(f"Error in pillar research for {pillar_name}: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "pillar": pillar_name
            }

    async def research_and_score_city(
        self,
        city_name: str,
        city_address: str,
        evaluator_score: Optional[float] = None,
        aIScore: Optional[float] = None,
        pillars_context: Optional[str] = None,
        year: int = None
    ) -> Dict[str, Any]:
        """
        Conduct comprehensive city-level assessment
        
        Args:
            city_name: Name of the city
            city_address: Full address
            evaluator_score: Human evaluator's city score
            pillar_scores: Optional list of AI pillar scores
            year: Assessment year
        """
        try:
            await self._ensure_initialized()
            
            if year is None:
                year = datetime.now().year
            
            # Build pillar summary context
            pillars_context = "\n**PILLAR-LEVEL FINDINGS** (for synthesis):\n" + pillars_context
            
            
            prompt = ChatPromptTemplate.from_messages([
                ("system",VerdianPromptTemplates._get_city_system_prompt(self, city_name, city_address, year, evaluator_score, aIScore, pillars_context)),
                ("user", """Conduct comprehensive city-wide assessment:

                City: {city_name}
                Address: {city_address}
                Year: {year}
                aIScore:{aIScore}
                {evaluator_context}

                SEARCH THE WEB comprehensively for city-level data. Synthesize findings across all 14 pillars. Provide holistic Veridian Urban Index evaluation with clear evidence.""")
            ])

            evaluator_context = ""
            if evaluator_score is not None:
                evaluator_context = f"Evaluator's City Score: {evaluator_score}/100"

            for attempt in range(self.max_retries):
                try:
                    chain = prompt | self.llm | StrOutputParser()
                    
                    result = await chain.ainvoke({
                        "city_name": city_name,
                        "city_address": city_address,
                        "pillars_context": pillars_context,
                        "year": year,
                        "aIScore":aIScore if aIScore else "Not provided",
                        "evaluator_context": evaluator_context
                    })

                    if not result or result.strip() == "{}":
                        continue  # retry

                     # Parse and validate                   

                    analysis = json.loads(jrp.clean_json_response(result))
                    jrp.validate_city_response(analysis)                    
                  
                    discrepancy = None
                    if evaluator_score is not None and analysis.get("ai_progress") is not None:
                        discrepancy = jrp._calculate_discrepancy(
                            analysis.get("ai_progress"),
                            evaluator_score
                        )                    
                    mapped = jrp.map_city_response(
                        analysis=analysis,
                        city_name=city_name,
                        year=year,
                        discrepancy=discrepancy
                    )

                    return mapped
                except (json.JSONDecodeError, ValueError) as e:
                    logger.error(f"JSON parse error on attempt {attempt + 1}: {e}")
                    if attempt < self.max_retries - 1:
                        await asyncio.sleep(self.retry_delay)
                        continue
                    else:
                        raise
        except Exception as e:
            logger.error(f"Error in city research: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    async def immediate_situation(
    self,
    city_name: str,
    country: str,
    ai_city_context: str,
    documentContext: Optional[str],
    year: int = None,
    ) -> Dict[str, Any]:
        """Produce a cross-pillar city-level urban assessment."""

        try:
            # Decide which prompt to use
            if not documentContext or len(documentContext.strip()) < 100:
                pillar_names = PillarPrompts.get_all_pillar_names()

                pillar_list_str = "\n".join(
                    f"{k}. {v}" for k, v in pillar_names.items()
                )

                system_prompt = (
                    VerdianPromptTemplates.city_situation_awareness_system_prompt(
                        pillar_list_str
                    )
                )
            else:
                system_prompt = (
                    VerdianPromptTemplates.city_summery_system_prompt(
                        publicContext=ai_city_context,
                        documentContext=documentContext,
                    )
                )

            label = f"city|{city_name}"

            user_template = """
    City: {city_name}
    Country: {country}
    Year: {year}
    """

            raw = await self._llm_svc.invoke_chain(
                system_prompt=system_prompt,
                user_template=user_template,
                variables={
                    "city_name": city_name,
                    "country": country,
                    "year": year,
                },
                label=label,
            )

            analysis = json.loads(jrp.clean_json_response(raw))

            return jrp.build_immediateSituation_record(analysis)

        except Exception as exc:
            logger.exception("immediate_situation failed")
            return {
                "success": False,
                "error": str(exc),
            }
        

# Singleton instance
veridian_ai_research_service = VerdianAIResearchService()