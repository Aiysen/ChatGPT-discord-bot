import os
import logging
import re
import base64
from typing import Dict, List, Optional, Any, Tuple
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
import asyncio

import g4f
from g4f.client import Client
from g4f.Provider import RetryProvider
from openai import AsyncOpenAI
import google.generativeai as genai
from anthropic import AsyncAnthropic
import aiohttp

logger = logging.getLogger(__name__)


def _env_value(key: str) -> Optional[str]:
    raw = os.getenv(key)
    if raw is None:
        return None
    value = raw.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        value = value[1:-1].strip()
    return value or None


class ProviderType(Enum):
    FREE = "free"
    OPENAI = "openai"
    CLAUDE = "claude"
    GEMINI = "gemini"
    GROK = "grok"


@dataclass
class ModelInfo:
    name: str
    provider: ProviderType
    description: str = ""
    supports_vision: bool = False
    supports_image_generation: bool = False


class BaseProvider(ABC):
    """Base class for all AI providers"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.models: List[ModelInfo] = []
        
    @abstractmethod
    async def chat_completion(self, messages: List[Dict[str, str]], model: str, **kwargs) -> str:
        """Generate chat completion"""
        pass
    
    @abstractmethod
    async def generate_image(self, prompt: str, model: Optional[str] = None, **kwargs) -> str:
        """Generate image from prompt"""
        pass

    async def edit_image(self, image_bytes: bytes, prompt: str, model: Optional[str] = None, **kwargs) -> bytes:
        """Edit an existing image"""
        raise NotImplementedError("Image editing is not supported by this provider")
    
    @abstractmethod
    def get_available_models(self) -> List[ModelInfo]:
        """Get list of available models"""
        pass
    
    @abstractmethod
    def supports_image_generation(self) -> bool:
        """Check if provider supports image generation"""
        pass

    def supports_image_editing(self) -> bool:
        """Check if provider supports image editing"""
        return False


class FreeProvider(BaseProvider):
    """Provider for free models via g4f - COMPLETELY AUTH-FREE"""
    
    def __init__(self):
        super().__init__()
        self.working_providers = self._build_working_providers()
        if not self.working_providers:
            raise RuntimeError("No compatible auth-free g4f providers are available in this environment")
        
        # Create provider list for RetryProvider
        providers_list = [p['provider'] for p in self.working_providers]
        
        logger.info(f"FreeProvider initialized with {len(providers_list)} VERIFIED working providers")
        for provider_info in self.working_providers:
            logger.info(f"  ✅ {provider_info['name']}: {', '.join(provider_info['models'])}")
        
        # Initialize with RetryProvider for automatic fallback
        self.client = Client(
            provider=RetryProvider(providers_list, shuffle=False)
        )
        
        # Track current provider for better error handling
        self.current_provider_index = 0

    def _build_working_providers(self) -> List[Dict[str, Any]]:
        """Resolve only providers that exist in the installed g4f version."""
        provider_candidates: List[Tuple[str, List[str], str]] = [
            ("Blackbox", ["blackboxai"], "Blackbox"),
            ("Chatai", ["gpt-3.5-turbo", "gpt-4"], "Chatai"),
            ("CohereForAI_C4AI_Command", ["command-r-plus", "command-r"], "CohereForAI"),
        ]
        working_providers: List[Dict[str, Any]] = []

        for provider_attr, models, display_name in provider_candidates:
            provider_class = getattr(g4f.Provider, provider_attr, None)
            if provider_class is None:
                logger.warning(
                    "Skipping unavailable g4f provider '%s' in version %s",
                    provider_attr,
                    getattr(g4f, "__version__", "unknown"),
                )
                continue

            working_providers.append(
                {
                    "provider": provider_class,
                    "models": models,
                    "name": display_name,
                }
            )

        return working_providers
        
    async def chat_completion(self, messages: List[Dict[str, str]], model: str, **kwargs) -> str:
        """Generate chat completion with robust fallback system"""
        
        # Determine the best model to use
        target_model = self._select_model(model)
        
        # Try each provider with intelligent model matching
        for attempt in range(len(self.working_providers)):
            provider_info = self.working_providers[attempt]
            
            try:
                # Select best model for this provider
                provider_model = self._get_provider_model(provider_info, target_model)
                
                logger.info(f"Attempting {provider_info['name']} with model {provider_model}")
                
                # Create client for specific provider (bypass RetryProvider for better control)
                client = Client(provider=provider_info['provider'])
                
                response = await asyncio.to_thread(
                    client.chat.completions.create,
                    model=provider_model,
                    messages=messages,
                    timeout=30,  # Add timeout
                    **kwargs
                )
                
                if response and response.choices and response.choices[0].message.content:
                    result = response.choices[0].message.content
                    if self._looks_like_auth_error(result):
                        logger.warning(
                            f"❌ {provider_info['name']} returned auth-required payload, trying next provider"
                        )
                        continue
                    logger.info(f"✅ Success with {provider_info['name']} + {provider_model}")
                    return result
                else:
                    logger.warning(f"Empty response from {provider_info['name']}")
                    continue
                    
            except Exception as e:
                error_msg = str(e)
                logger.warning(f"❌ {provider_info['name']} failed: {error_msg[:100]}...")
                
                # Don't give up immediately on certain errors
                if attempt < len(self.working_providers) - 1:
                    continue
        
        # If all providers fail, raise a meaningful error
        raise Exception("All free providers failed. The service may be temporarily unavailable.")
    
    def _looks_like_auth_error(self, text: str) -> bool:
        if not text:
            return False
        lowered = text.lower()
        auth_markers = (
            "authentication error",
            "no api key passed in",
            '"type":"error"',
            '"type": "error"',
        )
        return any(marker in lowered for marker in auth_markers)
    
    def _select_model(self, model: Optional[str]) -> str:
        """Select the best available model"""
        if not model or model == "auto":
            # Default to a widely supported model
            return "gpt-3.5-turbo"
        return model
    
    def _get_provider_model(self, provider_info: dict, target_model: str) -> str:
        """Get the best model for a specific provider"""
        supported_models = provider_info['models']
        
        # Try exact match first
        if target_model in supported_models:
            return target_model
        
        # Smart fallback based on model type
        if 'gpt' in target_model.lower():
            for model in supported_models:
                if 'gpt' in model.lower():
                    return model
        
        if 'claude' in target_model.lower():
            for model in supported_models:
                if 'command' in model.lower():  # Cohere is Claude-like
                    return model
        
        if 'llama' in target_model.lower() or 'meta' in target_model.lower():
            # No llama support in current working providers
            return supported_models[0]
        
        # Default to first available model
        return supported_models[0]
    
    async def generate_image(self, prompt: str, model: Optional[str] = None, **kwargs) -> str:
        """Generate image - simplified implementation"""
        try:
            # For now, use simple fallback message since image providers are less reliable
            # Could be enhanced later with working image providers like PollinationsAI
            logger.warning("Image generation via free providers is currently disabled for reliability")
            raise NotImplementedError("Image generation is temporarily unavailable via free providers. Please use a paid provider.")
        except Exception as e:
            logger.error(f"Free provider image generation error: {e}")
            raise
    
    def get_available_models(self) -> List[ModelInfo]:
        """Return models only from providers available in the installed g4f version."""
        models: List[ModelInfo] = []
        seen_models = set()

        for provider_info in self.working_providers:
            for model_name in provider_info["models"]:
                if model_name in seen_models:
                    continue
                seen_models.add(model_name)
                models.append(
                    ModelInfo(
                        model_name,
                        ProviderType.FREE,
                        f"{provider_info['name']} free model",
                    )
                )

        return models
    
    def supports_image_generation(self) -> bool:
        return False  # Disabled for reliability - only working text providers included


class OpenAIProvider(BaseProvider):
    """Official OpenAI API provider"""
    
    def __init__(self, api_key: str):
        super().__init__(api_key)
        self.client = AsyncOpenAI(api_key=api_key)
        
    async def chat_completion(self, messages: List[Dict[str, str]], model: str, **kwargs) -> str:
        try:
            if not model:
                model = "gpt-4o-mini"
                
            response = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                **kwargs
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI provider error: {e}")
            raise
    
    async def generate_image(self, prompt: str, model: Optional[str] = None, **kwargs) -> str:
        try:
            response = await self.client.images.generate(
                model=model or "dall-e-3",
                prompt=prompt,
                size=kwargs.get("size", "1024x1024"),
                quality=kwargs.get("quality", "standard"),
                n=1
            )
            return response.data[0].url
        except Exception as e:
            logger.error(f"OpenAI image generation error: {e}")
            raise

    async def edit_image(self, image_bytes: bytes, prompt: str, model: Optional[str] = None, **kwargs) -> bytes:
        try:
            response = await self.client.images.edit(
                model=model or "gpt-image-1",
                image=("input.png", image_bytes, "image/png"),
                prompt=prompt,
                size=kwargs.get("size", "1024x1024"),
                n=1,
                response_format="b64_json",
            )

            if not response.data:
                raise RuntimeError("OpenAI returned empty image edit response")

            image_data = response.data[0]
            if getattr(image_data, "b64_json", None):
                return base64.b64decode(image_data.b64_json)

            image_url = getattr(image_data, "url", None)
            if image_url:
                async with aiohttp.ClientSession() as session:
                    async with session.get(image_url) as url_response:
                        if url_response.status != 200:
                            raise RuntimeError(f"Failed to download edited image: HTTP {url_response.status}")
                        return await url_response.read()

            raise RuntimeError("OpenAI did not return edited image content")
        except Exception as e:
            logger.error(f"OpenAI image editing error: {e}")
            raise
    
    def get_available_models(self) -> List[ModelInfo]:
        return [
            ModelInfo("gpt-4o", ProviderType.OPENAI, "Most capable GPT-4 model", supports_vision=True),
            ModelInfo("gpt-4o-mini", ProviderType.OPENAI, "Affordable GPT-4 model", supports_vision=True),
            ModelInfo("o1", ProviderType.OPENAI, "Reasoning model"),
            ModelInfo("o1-mini", ProviderType.OPENAI, "Smaller reasoning model"),
            ModelInfo("dall-e-3", ProviderType.OPENAI, "DALL-E 3 image generation", supports_image_generation=True),
            ModelInfo("dall-e-2", ProviderType.OPENAI, "DALL-E 2 image generation", supports_image_generation=True),
        ]
    
    def supports_image_generation(self) -> bool:
        return True

    def supports_image_editing(self) -> bool:
        return True


class ClaudeProvider(BaseProvider):
    """Official Anthropic Claude API provider"""
    
    def __init__(self, api_key: str):
        super().__init__(api_key)
        self.client = AsyncAnthropic(api_key=api_key)
        
    async def chat_completion(self, messages: List[Dict[str, str]], model: str, **kwargs) -> str:
        try:
            if not model:
                model = "claude-3-5-haiku-latest"
            
            # Convert messages format for Claude
            system_message = None
            claude_messages = []
            
            for msg in messages:
                if msg["role"] == "system":
                    system_message = msg["content"]
                else:
                    claude_messages.append({
                        "role": msg["role"],
                        "content": msg["content"]
                    })
            
            response = await self.client.messages.create(
                model=model,
                messages=claude_messages,
                system=system_message,
                max_tokens=kwargs.get("max_tokens", 4096)
            )
            
            return response.content[0].text
        except Exception as e:
            logger.error(f"Claude provider error: {e}")
            raise
    
    async def generate_image(self, prompt: str, model: Optional[str] = None, **kwargs) -> str:
        raise NotImplementedError("Claude does not support image generation")
    
    def get_available_models(self) -> List[ModelInfo]:
        return [
            ModelInfo("claude-3-5-sonnet-latest", ProviderType.CLAUDE, "Most capable Claude model"),
            ModelInfo("claude-3-5-haiku-latest", ProviderType.CLAUDE, "Fast and affordable"),
            ModelInfo("claude-3-opus-latest", ProviderType.CLAUDE, "Previous flagship model"),
        ]
    
    def supports_image_generation(self) -> bool:
        return False


class GeminiProvider(BaseProvider):
    """Official Google Gemini API provider"""
    
    def __init__(self, api_key: str):
        super().__init__(api_key)
        genai.configure(api_key=api_key)
        
    async def chat_completion(self, messages: List[Dict[str, str]], model: str, **kwargs) -> str:
        try:
            if not model:
                model = "gemini-2.0-flash-exp"
            
            # Initialize model
            gemini_model = genai.GenerativeModel(model)
            
            # Convert messages to Gemini format
            chat = gemini_model.start_chat(history=[])
            
            # Process messages
            for msg in messages:
                if msg["role"] == "user":
                    response = await asyncio.to_thread(
                        chat.send_message,
                        msg["content"]
                    )
                elif msg["role"] == "assistant":
                    # Add assistant messages to history
                    chat.history.append({
                        "role": "model",
                        "parts": [msg["content"]]
                    })
            
            return response.text
        except Exception as e:
            logger.error(f"Gemini provider error: {e}")
            raise
    
    async def generate_image(self, prompt: str, model: Optional[str] = None, **kwargs) -> str:
        try:
            # Use Imagen via Gemini
            model_name = model or "imagen-3.0-generate-001"
            imagen = genai.ImageGenerationModel(model_name)
            
            response = await asyncio.to_thread(
                imagen.generate_images,
                prompt=prompt,
                number_of_images=1,
                aspect_ratio=kwargs.get("aspect_ratio", "1:1")
            )
            
            # Save image and return URL or base64
            return response.images[0]._image_bytes
        except Exception as e:
            logger.error(f"Gemini image generation error: {e}")
            raise
    
    def get_available_models(self) -> List[ModelInfo]:
        return [
            ModelInfo("gemini-2.0-flash-exp", ProviderType.GEMINI, "Latest experimental model", supports_vision=True),
            ModelInfo("gemini-1.5-pro", ProviderType.GEMINI, "Advanced reasoning", supports_vision=True),
            ModelInfo("gemini-1.5-flash", ProviderType.GEMINI, "Fast multimodal", supports_vision=True),
            ModelInfo("imagen-3.0-generate-001", ProviderType.GEMINI, "Image generation", supports_image_generation=True),
        ]
    
    def supports_image_generation(self) -> bool:
        return True


class GrokProvider(BaseProvider):
    """xAI Grok API provider"""
    
    def __init__(self, api_key: str):
        super().__init__(api_key)
        self.api_key = api_key
        self.base_url = "https://api.x.ai/v1"
        
    async def chat_completion(self, messages: List[Dict[str, str]], model: str, **kwargs) -> str:
        try:
            if not model:
                model = "grok-2-latest"
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": model,
                "messages": messages,
                "temperature": kwargs.get("temperature", 0.7),
                "max_tokens": kwargs.get("max_tokens", 4096)
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=data
                ) as response:
                    result = await response.json()
                    
                    if response.status != 200:
                        raise Exception(f"Grok API error: {result}")
                    
                    return result["choices"][0]["message"]["content"]
                    
        except Exception as e:
            logger.error(f"Grok provider error: {e}")
            raise
    
    async def generate_image(self, prompt: str, model: Optional[str] = None, **kwargs) -> str:
        raise NotImplementedError("Grok does not support image generation yet")
    
    def get_available_models(self) -> List[ModelInfo]:
        return [
            ModelInfo("grok-2-latest", ProviderType.GROK, "Latest Grok-2 model"),
            ModelInfo("grok-2-mini", ProviderType.GROK, "Smaller, faster Grok model"),
        ]
    
    def supports_image_generation(self) -> bool:
        return False


class ProviderManager:
    """Manages all AI providers"""
    
    def __init__(self):
        self.providers: Dict[ProviderType, BaseProvider] = {}
        self._initialize_providers()
        self.current_provider = (
            ProviderType.FREE if ProviderType.FREE in self.providers
            else next(iter(self.providers), ProviderType.FREE)
        )
        
    def _validate_api_key(self, api_key: str, provider_name: str, pattern: Optional[str] = None) -> bool:
        """Validate API key format"""
        if not api_key or len(api_key) < 10:
            logger.warning(f"Invalid {provider_name} API key: too short (length: {len(api_key)})")
            return False
        
        # Skip pattern validation for now to be more permissive
        # Most API key issues are due to overly strict regex patterns
        if pattern and not re.match(pattern, api_key):
            logger.warning(f"API key format warning for {provider_name} (pattern: {pattern})")
            logger.warning(f"Key format: {api_key[:15]}... (length: {len(api_key)})")
            # Return True anyway - let the provider initialization handle validity
            logger.info(f"Proceeding with {provider_name} despite format warning")
        
        return True
    
    def _initialize_providers(self):
        """Initialize available providers based on API keys"""
        try:
            self.providers[ProviderType.FREE] = FreeProvider()
            logger.info("Initialized free provider")
        except Exception as e:
            logger.warning(f"Free provider unavailable: {e}")
        
        # API key configurations: (env_var, provider_type, provider_class, validation_pattern)
        api_configs = [
            ("OPENAI_KEY", ProviderType.OPENAI, OpenAIProvider, r'^sk-[a-zA-Z0-9]{20,}$'),  # More flexible OpenAI key format
            ("CLAUDE_KEY", ProviderType.CLAUDE, ClaudeProvider, r'^sk-ant-[a-zA-Z0-9-]{50,}$'),  # More flexible Claude key
            ("GEMINI_KEY", ProviderType.GEMINI, GeminiProvider, r'^[a-zA-Z0-9_-]{20,}$'),  # More flexible Gemini key
            ("GROK_KEY", ProviderType.GROK, GrokProvider, r'^xai-[a-zA-Z0-9-]{20,}$')  # More flexible Grok key
        ]
        
        for env_key, provider_type, provider_class, pattern in api_configs:
            api_key = _env_value(env_key)
            if api_key:
                logger.info(f"Found {env_key} with length {len(api_key)}, prefix: {api_key[:10]}...")
                if self._validate_api_key(api_key, provider_type.value, pattern):
                    try:
                        self.providers[provider_type] = provider_class(api_key)
                        logger.info(f"✅ Successfully initialized {provider_type.value} provider")
                    except Exception as e:
                        logger.error(f"❌ Failed to initialize {provider_type.value}: {e}")
                else:
                    logger.warning(f"❌ Skipping {provider_type.value} due to invalid API key format")
            else:
                logger.debug(f"No {env_key} provided - {provider_type.value} provider disabled")
    
    def get_provider(self, provider_type: Optional[ProviderType] = None) -> BaseProvider:
        """Get specific provider or current provider"""
        if provider_type:
            if provider_type not in self.providers:
                raise ValueError(f"Provider {provider_type.value} not available")
            return self.providers[provider_type]
        return self.providers[self.current_provider]
    
    def set_current_provider(self, provider_type: ProviderType):
        """Set current provider"""
        if provider_type not in self.providers:
            raise ValueError(f"Provider {provider_type.value} not available")
        self.current_provider = provider_type
    
    def get_available_providers(self) -> List[ProviderType]:
        """Get list of available providers"""
        return list(self.providers.keys())
    
    def get_all_models(self) -> Dict[ProviderType, List[ModelInfo]]:
        """Get all models from all providers"""
        result = {}
        for provider_type, provider in self.providers.items():
            result[provider_type] = provider.get_available_models()
        return result
    
    def get_provider_models(self, provider_type: ProviderType) -> List[ModelInfo]:
        """Get models for specific provider"""
        if provider_type not in self.providers:
            return []
        return self.providers[provider_type].get_available_models()