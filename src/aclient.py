import os
import discord
import asyncio
import logging
from typing import List, Dict, Optional

from src import personas
from src.log import logger
from src.providers import ProviderManager, ProviderType, ModelInfo
from utils.message_utils import send_split_message

from dotenv import load_dotenv
from discord import app_commands

load_dotenv()


def _env_value(key: str, default: Optional[str] = None) -> Optional[str]:
    raw = os.getenv(key)
    if raw is None:
        return default
    value = raw.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        value = value[1:-1].strip()
    return value or default


def _env_truthy(key: str) -> bool:
    v = _env_value(key)
    return v is not None and v.lower() in ("1", "true", "yes", "on")


def _looks_like_openai_quota_error(error_text: str) -> bool:
    text = (error_text or "").lower()
    return (
        "insufficient_quota" in text
        or "exceeded your current quota" in text
        or "billing_hard_limit_reached" in text
    )


def _looks_like_openai_auth_error(error_text: str) -> bool:
    text = (error_text or "").lower()
    return (
        "invalid_api_key" in text
        or "incorrect api key" in text
        or "no api key provided" in text
        or "authenticationerror" in text
    )


class DiscordClient(discord.Client):
    def __init__(self) -> None:
        intents = discord.Intents.default()
        intents.message_content = _env_truthy("DISCORD_MESSAGE_CONTENT_INTENT")
        super().__init__(intents=intents)
        
        self.tree = app_commands.CommandTree(self)
        
        # Initialize provider manager
        self.provider_manager = ProviderManager()
        
        # Set default provider and model
        default_provider = _env_value("DEFAULT_PROVIDER", "free")
        try:
            self.provider_manager.set_current_provider(ProviderType(default_provider))
        except ValueError:
            fallback_provider = self._get_startup_provider_fallback()
            logger.warning(
                f"Default provider {default_provider} is unavailable, using {fallback_provider.value}"
            )
            self.provider_manager.set_current_provider(fallback_provider)
        
        self.current_model = _env_value("DEFAULT_MODEL", "auto")
        
        # Conversation management
        self.conversation_history = []
        self.current_channel = None
        self.current_persona = "standard"
        
        # Bot settings
        self.activity = discord.Activity(
            type=discord.ActivityType.listening, 
            name="/chat | /help | /provider"
        )
        self.isPrivate = False
        self.is_replying_all = _env_truthy("REPLYING_ALL")
        self.replying_all_discord_channel_id = _env_value("REPLYING_ALL_DISCORD_CHANNEL_ID")
        if self.is_replying_all and not intents.message_content:
            logger.warning(
                "REPLYING_ALL без DISCORD_MESSAGE_CONTENT_INTENT: включите Message Content Intent "
                "в портале Discord и задайте DISCORD_MESSAGE_CONTENT_INTENT=true"
            )
        
        # Load system prompt
        config_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        prompt_path = os.path.join(config_dir, "system_prompt.txt")
        try:
            with open(prompt_path, "r", encoding="utf-8") as f:
                self.starting_prompt = f.read()
        except FileNotFoundError:
            self.starting_prompt = ""
            logger.warning("system_prompt.txt not found")
        
        # Message queue for rate limiting
        self.message_queue = asyncio.Queue()

    def _get_startup_provider_fallback(self) -> ProviderType:
        """Return the best provider available for startup."""
        available_providers = self.provider_manager.get_available_providers()
        if not available_providers:
            raise RuntimeError("No AI providers are available. Configure at least one provider.")

        if ProviderType.FREE in available_providers:
            return ProviderType.FREE

        return available_providers[0]
    
    async def process_messages(self):
        """Process queued messages"""
        while True:
            if self.current_channel is not None:
                while not self.message_queue.empty():
                    async with self.current_channel.typing():
                        message, user_message = await self.message_queue.get()
                        try:
                            await self.send_message(message, user_message)
                        except Exception as e:
                            logger.exception(f"Error while processing message: {e}")
                        finally:
                            self.message_queue.task_done()
            await asyncio.sleep(1)
    
    async def enqueue_message(self, message, user_message):
        """Add message to processing queue"""
        await message.response.defer(ephemeral=self.isPrivate) if hasattr(message, 'response') else None
        await self.message_queue.put((message, user_message))
    
    async def send_message(self, message, user_message):
        """Send response to user"""
        if hasattr(message, 'user'):  # Slash command
            author = message.user.id
        else:  # Regular message
            author = message.author.id
        
        try:
            response = await self.handle_response(user_message)
            response_content = f'> **{user_message}** - <@{str(author)}> \n\n{response}'
            await send_split_message(self, response_content, message)
        except Exception as e:
            logger.exception(f"Error while sending: {e}")
            error_msg = f"❌ Error: {str(e)}"
            if hasattr(message, 'followup'):
                await message.followup.send(error_msg)
            else:
                await message.channel.send(error_msg)
    
    async def send_start_prompt(self):
        """Send initial system prompt"""
        discord_channel_id = _env_value("DISCORD_CHANNEL_ID")
        try:
            if self.starting_prompt and discord_channel_id:
                channel = self.get_channel(int(discord_channel_id))
                logger.info(f"Send system prompt with size {len(self.starting_prompt)}")
                
                response = await self.handle_response(self.starting_prompt)
                await channel.send(f"{response}")
                
                logger.info(f"System prompt response: {response}")
            else:
                logger.info("No starting prompt given or no Discord channel selected.")
        except Exception as e:
            logger.exception(f"Error while sending system prompt: {e}")
    
    async def handle_response(self, user_message: str) -> str:
        """Generate response using current provider"""
        # Add user message to history
        self.conversation_history.append({'role': 'user', 'content': user_message})
        
        # Better conversation management
        MAX_CONVERSATION_LENGTH = int(os.getenv("MAX_CONVERSATION_LENGTH", "20"))
        CONVERSATION_TRIM_SIZE = int(os.getenv("CONVERSATION_TRIM_SIZE", "8"))
        
        if len(self.conversation_history) > MAX_CONVERSATION_LENGTH:
            # Keep system prompts (first few messages) and recent context
            system_messages = [m for m in self.conversation_history[:3] if m['role'] == 'system']
            recent_messages = self.conversation_history[-CONVERSATION_TRIM_SIZE:]
            
            # Ensure we don't lose important context
            if system_messages:
                self.conversation_history = system_messages + recent_messages
            else:
                self.conversation_history = recent_messages
            
            logger.info(f"Trimmed conversation history to {len(self.conversation_history)} messages")
        
        # Get current provider
        provider = self.provider_manager.get_provider()
        
        try:
            # Generate response
            response = await provider.chat_completion(
                messages=self.conversation_history,
                model=self.current_model if self.current_model != "auto" else None
            )
            
            # Add to history
            self.conversation_history.append({'role': 'assistant', 'content': response})
            
            return response
            
        except Exception as e:
            logger.error(f"Provider error: {e}")
            error_text = str(e)

            if self.provider_manager.current_provider == ProviderType.OPENAI:
                if _looks_like_openai_quota_error(error_text):
                    error_response = (
                        "❌ OpenAI API: превышена квота (`insufficient_quota`). "
                        "Проверьте billing/лимиты и пополните баланс в OpenAI Platform."
                    )
                    self.conversation_history.append({'role': 'assistant', 'content': error_response})
                    return error_response

                if _looks_like_openai_auth_error(error_text):
                    error_response = (
                        "❌ OpenAI API: проблема с ключом доступа. "
                        "Проверьте `OPENAI_KEY` и что ключ активен в нужном проекте."
                    )
                    self.conversation_history.append({'role': 'assistant', 'content': error_response})
                    return error_response

            # Try fallback to free provider
            if self.provider_manager.current_provider != ProviderType.FREE:
                logger.info("Falling back to free provider")
                try:
                    free_provider = self.provider_manager.get_provider(ProviderType.FREE)
                    response = await free_provider.chat_completion(
                        messages=self.conversation_history,
                        model=None
                    )
                    self.conversation_history.append({'role': 'assistant', 'content': response})
                    return f"{response}\n\n*⚠️ Fallback to free provider due to error*"
                except Exception as fallback_error:
                    logger.error(f"Fallback provider also failed: {fallback_error}")
                    # Return user-friendly error message
                    error_response = "❌ I'm having trouble processing your request right now. Please try again later or contact an administrator."
                    self.conversation_history.append({'role': 'assistant', 'content': error_response})
                    return error_response
            else:
                # Already using free provider, return error
                error_response = "❌ The free provider is currently unavailable. Please try again later."
                self.conversation_history.append({'role': 'assistant', 'content': error_response})
                return error_response
    
    async def generate_image(self, prompt: str, model: Optional[str] = None) -> str:
        """Generate image using current provider"""
        provider = self.provider_manager.get_provider()
        
        if not provider.supports_image_generation():
            for provider_type in self.provider_manager.get_available_providers():
                candidate = self.provider_manager.get_provider(provider_type)
                if candidate.supports_image_generation():
                    provider = candidate
                    break
            else:
                raise NotImplementedError("No configured provider currently supports image generation")
        
        return await provider.generate_image(prompt, model)
    
    def reset_conversation_history(self):
        """Reset conversation and persona"""
        self.conversation_history = []
        self.current_persona = "standard"
        personas.current_persona = "standard"
    
    async def switch_persona(self, persona: str, user_id: Optional[str] = None) -> None:
        """Switch to a different persona"""
        self.reset_conversation_history()
        self.current_persona = persona
        personas.current_persona = persona
        
        # Add persona prompt to conversation (with permission check)
        persona_prompt = personas.get_persona_prompt(persona, user_id)
        self.conversation_history.append({'role': 'system', 'content': persona_prompt})
        
        # Get initial response with new persona
        await self.handle_response("Hello! Please confirm you understand your new role.")
    
    def get_current_provider_info(self) -> Dict:
        """Get information about current provider and model"""
        provider = self.provider_manager.get_provider()
        models = provider.get_available_models()
        
        return {
            "provider": self.provider_manager.current_provider.value,
            "current_model": self.current_model,
            "available_models": models,
            "supports_images": provider.supports_image_generation()
        }
    
    def switch_provider(self, provider_type: ProviderType, model: Optional[str] = None):
        """Switch to a different provider and optionally set model"""
        self.provider_manager.set_current_provider(provider_type)
        if model:
            self.current_model = model
        else:
            # Set to first available model or auto
            provider = self.provider_manager.get_provider()
            models = provider.get_available_models()
            self.current_model = models[0].name if models else "auto"


# Create singleton instance
discordClient = DiscordClient()