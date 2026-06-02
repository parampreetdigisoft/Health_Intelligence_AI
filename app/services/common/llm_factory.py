"""
LLM Provider Factory - Abstract factory pattern for multiple LLM providers
"""
import logging
from abc import ABC, abstractmethod
from typing import Optional
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from app.config import settings, LLMProvider

logger = logging.getLogger(__name__)


class LLMProviderInterface(ABC):
    """Abstract interface for LLM providers"""
    
    @abstractmethod
    def get_llm(self, **kwargs) -> BaseChatModel:
        """Get configured LLM instance"""
        pass
    
    @abstractmethod
    def get_model_name(self) -> str:
        """Get the model name being used"""
        pass

class OpenAIProvider(LLMProviderInterface):
    """OpenAI LLM Provider"""
    
    def get_llm(self, **kwargs) -> BaseChatModel:
        if not settings.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not configured")
        
        model = kwargs.get("model", settings.OPENAI_MODEL)
        temperature = kwargs.get("temperature", settings.OPENAI_TEMPERATURE)
        max_tokens = kwargs.get("max_tokens", settings.LLM_MAX_TOKENS)
        
        logger.info(f"Initializing OpenAI with model: {model}")
        
        return ChatOpenAI(
            api_key=settings.OPENAI_API_KEY,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    
    def get_model_name(self) -> str:
        return settings.OPENAI_MODEL

class OpenRouterProvider(LLMProviderInterface):
    """OpenRouter LLM Provider (uses OpenAI-compatible API)"""
    
    def get_llm(self, **kwargs) -> BaseChatModel:
        if not settings.OPENROUTER_API_KEY:
            raise ValueError("OPENROUTER_API_KEY not configured")
        
        model = kwargs.get("model", settings.OPENROUTER_MODEL)
        temperature = kwargs.get("temperature", settings.LLM_TEMPERATURE)
        max_tokens = kwargs.get("max_tokens", settings.LLM_MAX_TOKENS)
        
        logger.info(f"Initializing OpenRouter with model: {model}")
        
        return ChatOpenAI(
            api_key=settings.OPENROUTER_API_KEY,
            base_url=settings.OPENROUTER_BASE_URL,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            default_headers={
                "HTTP-Referer": "http://localhost:8000",
                "X-Title": "AI Microservice"
            }
        )
    
    def get_model_name(self) -> str:
        return settings.OPENROUTER_MODEL

class LLMFactory:
    """Factory class to create LLM instances based on configuration"""
    
    _providers = {
        LLMProvider.OPENAI: OpenAIProvider,
        LLMProvider.OPENROUTER: OpenRouterProvider,
    }
    
    @classmethod
    def create_llm(cls, provider: Optional[str] = None, **kwargs) -> BaseChatModel:
        """
        Create an LLM instance based on the provider
        
        Args:
            provider: LLM provider name (defaults to settings.LLM_PROVIDER)
            **kwargs: Additional arguments to pass to the LLM
        
        Returns:
            BaseChatModel: Configured LLM instance
        """
        provider_name = provider or settings.LLM_PROVIDER
        
        try:
            provider_enum = LLMProvider(provider_name.lower())
        except ValueError:
            raise ValueError(
                f"Unsupported LLM provider: {provider_name}. "
                f"Supported providers: {[p.value for p in LLMProvider]}"
            )
        
        provider_class = cls._providers.get(provider_enum)
        if not provider_class:
            raise ValueError(f"Provider {provider_name} not implemented")
        
        provider_instance = provider_class()
        return provider_instance.get_llm(**kwargs)
    
    @classmethod
    def get_current_provider_name(cls) -> str:
        """Get the name of the current LLM provider"""
        provider_name = settings.LLM_PROVIDER.lower()
        provider_enum = LLMProvider(provider_name)
        provider_class = cls._providers.get(provider_enum)
        
        if provider_class:
            provider_instance = provider_class()
            return provider_instance.get_model_name()
        
        return "Unknown"



# Singleton instance for easy access
llm_factory = LLMFactory()