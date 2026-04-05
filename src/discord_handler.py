import asyncio
import io
import os
import time
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple

import discord
from discord import app_commands
from PIL import Image

from src.history_store import GenerationRecord, HistoryStore
from src.image_generator import GeneratedImage, ImageGenerator
from src.log import logger
from src.prompt_enhancer import STYLE_PRESETS, PromptEnhancementResult, PromptEnhancer


class UserRateLimiter:
    def __init__(self, min_interval_seconds: int = 5, max_per_minute: int = 8):
        self.min_interval_seconds = min_interval_seconds
        self.max_per_minute = max_per_minute
        self._last_call = {}
        self._window = defaultdict(deque)

    def is_allowed(self, user_id: int) -> tuple[bool, Optional[str]]:
        now = time.monotonic()
        last_call = self._last_call.get(user_id, 0.0)
        if now - last_call < self.min_interval_seconds:
            return False, f"Слишком часто. Подождите {self.min_interval_seconds} сек."

        user_window = self._window[user_id]
        while user_window and now - user_window[0] > 60:
            user_window.popleft()
        if len(user_window) >= self.max_per_minute:
            return False, "Превышен лимит запросов в минуту."

        user_window.append(now)
        self._last_call[user_id] = now
        return True, None


class VariationSelect(discord.ui.Select):
    def __init__(
        self,
        workflow: "DiscordImageHandler",
        enhancement: PromptEnhancementResult,
        original_prompt: str,
        owner_id: int,
        had_input_image: bool,
        base_image_bytes: Optional[bytes] = None,
    ):
        self.workflow = workflow
        self.enhancement = enhancement
        self.original_prompt = original_prompt
        self.owner_id = owner_id
        self.had_input_image = had_input_image
        self.base_image_bytes = base_image_bytes

        options = []
        for idx, prompt in enumerate(enhancement.variations[:4]):
            label = f"Вариант {idx + 1}"
            description = prompt[:95] + ("..." if len(prompt) > 95 else "")
            options.append(discord.SelectOption(label=label, value=str(idx), description=description))

        super().__init__(
            placeholder="Выберите вариант промта для генерации",
            min_values=1,
            max_values=1,
            options=options,
        )

    async def callback(self, interaction: discord.Interaction):
        if interaction.user.id != self.owner_id:
            await interaction.response.send_message("❌ Этот селектор доступен только автору запроса.", ephemeral=True)
            return

        selected_index = int(self.values[0])
        selected_prompt = self.enhancement.variations[selected_index]
        await interaction.response.defer(thinking=True)

        try:
            if self.base_image_bytes:
                effective_prompt = self.workflow._build_edit_prompt(selected_prompt)
                images = await self.workflow.run_image_edit(effective_prompt, self.base_image_bytes)
                chosen_prompt = effective_prompt
                image_model = self.workflow.image_generator.edit_model
            else:
                images = await self.workflow.run_image_generation(selected_prompt)
                chosen_prompt = selected_prompt
                image_model = self.workflow.image_generator.model
            files = self.workflow.images_to_discord_files(images)
            self.workflow._remember_last_user_image(interaction.user.id, images)

            self.workflow.history_store.save(
                GenerationRecord(
                    user_id=interaction.user.id,
                    username=str(interaction.user),
                    original_prompt=self.original_prompt,
                    final_prompt=self.enhancement.final_prompt,
                    variations=self.enhancement.variations,
                    chosen_prompt=chosen_prompt,
                    prompt_model=self.enhancement.model,
                    image_model=image_model,
                    image_count=len(images),
                    had_input_image=self.base_image_bytes is not None,
                )
            )

            await interaction.followup.send(
                content=f"Сгенерировано по выбранному варианту:\n`{selected_prompt[:500]}`",
                files=files,
            )
        except Exception as exc:
            logger.exception(f"Variation generation failed: {exc}")
            await interaction.followup.send(f"❌ Ошибка генерации варианта: {exc}")


class VariationView(discord.ui.View):
    def __init__(
        self,
        workflow: "DiscordImageHandler",
        enhancement: PromptEnhancementResult,
        original_prompt: str,
        owner_id: int,
        had_input_image: bool,
        base_image_bytes: Optional[bytes] = None,
    ):
        super().__init__(timeout=600)
        self.add_item(
            VariationSelect(
                workflow=workflow,
                enhancement=enhancement,
                original_prompt=original_prompt,
                owner_id=owner_id,
                had_input_image=had_input_image,
                base_image_bytes=base_image_bytes,
            )
        )


class DiscordImageHandler:
    def __init__(self, discord_client: discord.Client):
        api_key = os.getenv("OPENAI_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_KEY is required for image commands (/imagine, /variations, /refine_last)")

        self.discord_client = discord_client
        self.prompt_enhancer = PromptEnhancer(api_key=api_key)
        self.image_generator = ImageGenerator(api_key=api_key)
        self.history_store = HistoryStore()
        self.rate_limiter = UserRateLimiter(
            min_interval_seconds=int(os.getenv("IMAGE_MIN_INTERVAL_SECONDS", "5")),
            max_per_minute=int(os.getenv("IMAGE_MAX_REQUESTS_PER_MINUTE", "8")),
        )
        self._worker_count = int(os.getenv("IMAGE_CONCURRENCY", "2"))
        self.task_timeout_seconds = int(os.getenv("IMAGE_TASK_TIMEOUT_SECONDS", "90"))
        self.default_image_count = int(os.getenv("IMAGINE_IMAGE_COUNT", "2"))
        self.default_edit_image_count = int(os.getenv("EDIT_IMAGE_COUNT", "1"))
        self._task_queue: asyncio.Queue = asyncio.Queue()
        self._workers: List[asyncio.Task] = []
        self._last_image_by_user: Dict[int, bytes] = {}

    def register_commands(self):
        @self.discord_client.tree.command(name="imagine", description="Сгенерировать изображение по промту")
        @app_commands.describe(
            prompt="Текстовый запрос",
            image="Исходное изображение (опционально)",
            style_preset="Стиль: ancient, fantasy, realistic (опционально)",
        )
        async def imagine(
            interaction: discord.Interaction,
            prompt: str,
            image: Optional[discord.Attachment] = None,
            style_preset: Optional[str] = None,
        ):
            await self.handle_imagine(interaction=interaction, prompt=prompt, image=image, style_preset=style_preset)

        @self.discord_client.tree.command(name="variations", description="Сгенерировать новые вариации последнего изображения")
        async def variations(interaction: discord.Interaction):
            await self.handle_variations(interaction)

        @self.discord_client.tree.command(
            name="refine_last",
            description="Доработать последний результат с дополнительным текстовым промтом",
        )
        @app_commands.describe(prompt="Что именно нужно доработать в последнем результате")
        async def refine_last(interaction: discord.Interaction, prompt: str):
            await self.handle_refine_last(interaction=interaction, prompt=prompt)

    async def handle_imagine(
        self,
        interaction: discord.Interaction,
        prompt: str,
        image: Optional[discord.Attachment] = None,
        style_preset: Optional[str] = None,
    ):
        clean_prompt = (prompt or "").strip()
        if not clean_prompt:
            await interaction.response.send_message("❌ Укажите prompt.", ephemeral=True)
            return
        if len(clean_prompt) > 1000:
            await interaction.response.send_message("❌ Prompt слишком длинный (макс. 1000 символов).", ephemeral=True)
            return
        if style_preset and style_preset.lower() not in STYLE_PRESETS:
            available = ", ".join(sorted(STYLE_PRESETS.keys()))
            await interaction.response.send_message(
                f"❌ Неизвестный style_preset. Доступно: {available}",
                ephemeral=True,
            )
            return

        allowed, error_text = self.rate_limiter.is_allowed(interaction.user.id)
        if not allowed:
            await interaction.response.send_message(f"❌ {error_text}", ephemeral=True)
            return

        image_bytes = None
        if image:
            if image.content_type and not image.content_type.startswith("image/"):
                await interaction.response.send_message("❌ Attachment должен быть изображением.", ephemeral=True)
                return
            image_bytes = await image.read()
            if not image_bytes:
                await interaction.response.send_message("❌ Пустой файл изображения.", ephemeral=True)
                return

        await interaction.response.defer(thinking=True)
        try:
            enhancement = await self.run_prompt_enhancement(
                user_prompt=clean_prompt,
                image_bytes=image_bytes,
                style_preset=style_preset,
            )
            images = await self.run_image_generation(enhancement.final_prompt)
            files = self.images_to_discord_files(images)
            self._remember_last_user_image(interaction.user.id, images)

            self.history_store.save(
                GenerationRecord(
                    user_id=interaction.user.id,
                    username=str(interaction.user),
                    original_prompt=clean_prompt,
                    final_prompt=enhancement.final_prompt,
                    variations=enhancement.variations,
                    chosen_prompt=enhancement.final_prompt,
                    prompt_model=enhancement.model,
                    image_model=self.image_generator.model,
                    image_count=len(images),
                    had_input_image=image_bytes is not None,
                )
            )

            view = VariationView(
                workflow=self,
                enhancement=enhancement,
                original_prompt=clean_prompt,
                owner_id=interaction.user.id,
                had_input_image=image_bytes is not None,
            )
            await interaction.followup.send(
                content=(
                    f"Использован улучшенный prompt:\n`{enhancement.final_prompt[:700]}`\n\n"
                    f"Модель улучшения: `{enhancement.model}`\n"
                    f"Модель генерации: `{self.image_generator.model}`"
                ),
                files=files,
                view=view,
            )
        except asyncio.TimeoutError:
            await interaction.followup.send("❌ Таймаут обработки запроса. Попробуйте снова.")
        except Exception as exc:
            logger.exception(f"/imagine failed: {exc}")
            await interaction.followup.send(f"❌ Ошибка генерации: {exc}")

    async def handle_draw(self, interaction: discord.Interaction, prompt: str):
        clean_prompt = (prompt or "").strip()
        if not clean_prompt:
            await interaction.response.send_message("❌ Укажите prompt.", ephemeral=True)
            return
        if len(clean_prompt) > 1000:
            await interaction.response.send_message("❌ Prompt слишком длинный (макс. 1000 символов).", ephemeral=True)
            return

        allowed, error_text = self.rate_limiter.is_allowed(interaction.user.id)
        if not allowed:
            await interaction.response.send_message(f"❌ {error_text}", ephemeral=True)
            return

        await interaction.response.defer(thinking=True)
        try:
            draw_model = "gpt-image-1.5"
            images = await self.run_image_generation(clean_prompt, model=draw_model)
            files = self.images_to_discord_files(images)
            self._remember_last_user_image(interaction.user.id, images)

            self.history_store.save(
                GenerationRecord(
                    user_id=interaction.user.id,
                    username=str(interaction.user),
                    original_prompt=clean_prompt,
                    final_prompt=clean_prompt,
                    variations=[clean_prompt],
                    chosen_prompt=clean_prompt,
                    prompt_model="none",
                    image_model=draw_model,
                    image_count=len(images),
                    had_input_image=False,
                )
            )

            await interaction.followup.send(
                content=(
                    f"Сгенерировано без улучшения промта:\n`{clean_prompt[:700]}`\n\n"
                    f"Модель генерации: `{draw_model}`"
                ),
                files=files,
            )
        except asyncio.TimeoutError:
            await interaction.followup.send("❌ Таймаут обработки запроса. Попробуйте снова.")
        except Exception as exc:
            logger.exception(f"/draw failed: {exc}")
            await interaction.followup.send(f"❌ Ошибка генерации: {exc}")

    async def handle_skill_ico(self, interaction: discord.Interaction, prompt: str):
        clean_prompt = (prompt or "").strip()
        if not clean_prompt:
            await interaction.response.send_message("❌ Укажите prompt.", ephemeral=True)
            return
        if len(clean_prompt) > 1000:
            await interaction.response.send_message("❌ Prompt слишком длинный (макс. 1000 символов).", ephemeral=True)
            return

        allowed, error_text = self.rate_limiter.is_allowed(interaction.user.id)
        if not allowed:
            await interaction.response.send_message(f"❌ {error_text}", ephemeral=True)
            return

        await interaction.response.defer(thinking=True)
        try:
            skill_model = "gpt-image-1.5"
            effective_prompt = self._build_skill_icon_prompt(clean_prompt)
            images = await self.run_image_generation(effective_prompt, model=skill_model, image_count=1)
            icon_images = self._convert_images_to_skill_icons(images)
            files = self.images_to_discord_files(icon_images, filename_prefix="skill_ico")
            self._remember_last_user_image(interaction.user.id, icon_images)

            self.history_store.save(
                GenerationRecord(
                    user_id=interaction.user.id,
                    username=str(interaction.user),
                    original_prompt=clean_prompt,
                    final_prompt=effective_prompt,
                    variations=[effective_prompt],
                    chosen_prompt=effective_prompt,
                    prompt_model="none",
                    image_model=skill_model,
                    image_count=len(icon_images),
                    had_input_image=False,
                )
            )

            await interaction.followup.send(
                content=(
                    f"Иконка навыка сгенерирована:\n`{clean_prompt[:700]}`\n\n"
                    "Формат: `100x100`, фон: `black`, без прозрачности."
                ),
                files=files,
            )
        except asyncio.TimeoutError:
            await interaction.followup.send("❌ Таймаут обработки запроса. Попробуйте снова.")
        except Exception as exc:
            logger.exception(f"/skill_ico failed: {exc}")
            await interaction.followup.send(f"❌ Ошибка генерации skill_ico: {exc}")

    async def handle_editimage(self, interaction: discord.Interaction, image: discord.Attachment, prompt: str):
        clean_prompt = (prompt or "").strip()
        if not clean_prompt:
            await interaction.response.send_message("❌ Укажите prompt.", ephemeral=True)
            return
        if len(clean_prompt) > 1000:
            await interaction.response.send_message("❌ Prompt слишком длинный (макс. 1000 символов).", ephemeral=True)
            return
        if image.content_type and not image.content_type.startswith("image/"):
            await interaction.response.send_message("❌ Attachment должен быть изображением.", ephemeral=True)
            return

        allowed, error_text = self.rate_limiter.is_allowed(interaction.user.id)
        if not allowed:
            await interaction.response.send_message(f"❌ {error_text}", ephemeral=True)
            return

        image_bytes = await image.read()
        if not image_bytes:
            await interaction.response.send_message("❌ Пустой файл изображения.", ephemeral=True)
            return

        await interaction.response.defer(thinking=True)
        try:
            effective_prompt = self._build_edit_prompt(clean_prompt)
            images = await self.run_image_edit(effective_prompt, image_bytes)
            files = self.images_to_discord_files(images)
            self._remember_last_user_image(interaction.user.id, images)

            self.history_store.save(
                GenerationRecord(
                    user_id=interaction.user.id,
                    username=str(interaction.user),
                    original_prompt=clean_prompt,
                    final_prompt=effective_prompt,
                    variations=[clean_prompt, effective_prompt],
                    chosen_prompt=effective_prompt,
                    prompt_model="none",
                    image_model=self.image_generator.edit_model,
                    image_count=len(images),
                    had_input_image=True,
                )
            )

            await interaction.followup.send(
                content=(
                    f"Изображение отредактировано по вашему prompt:\n`{clean_prompt[:700]}`\n\n"
                    f"Вариантов: `{len(images)}`\n"
                    f"Модель редактирования: `{self.image_generator.edit_model}`"
                ),
                files=files,
            )
        except asyncio.TimeoutError:
            await interaction.followup.send("❌ Таймаут редактирования изображения. Попробуйте снова.")
        except Exception as exc:
            logger.exception(f"/editimage failed: {exc}")
            await interaction.followup.send(f"❌ Ошибка редактирования изображения: {exc}")

    async def handle_variations(self, interaction: discord.Interaction):
        record = self.history_store.get_last(interaction.user.id)
        if not record:
            await interaction.response.send_message("❌ Нет истории. Сначала используйте `/imagine`.", ephemeral=True)
            return

        allowed, error_text = self.rate_limiter.is_allowed(interaction.user.id)
        if not allowed:
            await interaction.response.send_message(f"❌ {error_text}", ephemeral=True)
            return

        await interaction.response.defer(thinking=True)
        try:
            last_image_bytes = self._last_image_by_user.get(interaction.user.id)
            request_text = (
                "Create 4 meaningful prompt variations for regenerating the previous result. "
                "Keep subject and composition close, vary rendering approach, materials and lighting, "
                "and preserve recognizable identity of the last output:\n"
                f"{record.chosen_prompt}"
            )
            enhancement = await self.run_prompt_enhancement(
                user_prompt=request_text,
                image_bytes=last_image_bytes,
                style_preset=None,
            )
            selected_prompt = enhancement.variations[0]
            if last_image_bytes:
                effective_prompt = self._build_edit_prompt(selected_prompt)
                images = await self.run_image_edit(effective_prompt, last_image_bytes)
                saved_chosen_prompt = effective_prompt
                image_model = self.image_generator.edit_model
            else:
                images = await self.run_image_generation(selected_prompt)
                saved_chosen_prompt = selected_prompt
                image_model = self.image_generator.model
            files = self.images_to_discord_files(images)
            self._remember_last_user_image(interaction.user.id, images)

            self.history_store.save(
                GenerationRecord(
                    user_id=interaction.user.id,
                    username=str(interaction.user),
                    original_prompt=record.original_prompt,
                    final_prompt=record.final_prompt,
                    variations=enhancement.variations,
                    chosen_prompt=saved_chosen_prompt,
                    prompt_model=enhancement.model,
                    image_model=image_model,
                    image_count=len(images),
                    had_input_image=last_image_bytes is not None,
                )
            )

            view = VariationView(
                workflow=self,
                enhancement=enhancement,
                original_prompt=record.original_prompt,
                owner_id=interaction.user.id,
                had_input_image=last_image_bytes is not None,
                base_image_bytes=last_image_bytes,
            )
            await interaction.followup.send(
                content=(
                    "Сгенерированы новые вариации на базе последнего результата.\n"
                    f"Базовый вариант:\n`{selected_prompt[:700]}`"
                ),
                files=files,
                view=view,
            )
        except asyncio.TimeoutError:
            await interaction.followup.send("❌ Таймаут генерации вариаций.")
        except Exception as exc:
            logger.exception(f"/variations failed: {exc}")
            await interaction.followup.send(f"❌ Ошибка вариаций: {exc}")

    async def handle_refine_last(self, interaction: discord.Interaction, prompt: str):
        clean_prompt = (prompt or "").strip()
        if not clean_prompt:
            await interaction.response.send_message("❌ Укажите prompt с доработками.", ephemeral=True)
            return
        if len(clean_prompt) > 1000:
            await interaction.response.send_message("❌ Prompt слишком длинный (макс. 1000 символов).", ephemeral=True)
            return

        record = self.history_store.get_last(interaction.user.id)
        if not record:
            await interaction.response.send_message("❌ Нет истории. Сначала используйте `/imagine`.", ephemeral=True)
            return

        last_image_bytes = self._last_image_by_user.get(interaction.user.id)
        if not last_image_bytes:
            await interaction.response.send_message(
                "❌ Не найдено последнее изображение в памяти. Сначала сгенерируйте новую картинку и повторите.",
                ephemeral=True,
            )
            return

        allowed, error_text = self.rate_limiter.is_allowed(interaction.user.id)
        if not allowed:
            await interaction.response.send_message(f"❌ {error_text}", ephemeral=True)
            return

        await interaction.response.defer(thinking=True)
        try:
            effective_prompt = self._build_edit_prompt(clean_prompt)
            images = await self.run_image_edit(effective_prompt, last_image_bytes)
            files = self.images_to_discord_files(images)
            self._remember_last_user_image(interaction.user.id, images)

            self.history_store.save(
                GenerationRecord(
                    user_id=interaction.user.id,
                    username=str(interaction.user),
                    original_prompt=record.original_prompt,
                    final_prompt=effective_prompt,
                    variations=[clean_prompt, effective_prompt],
                    chosen_prompt=effective_prompt,
                    prompt_model="none",
                    image_model=self.image_generator.edit_model,
                    image_count=len(images),
                    had_input_image=True,
                )
            )
            await interaction.followup.send(
                content=(
                    f"Доработка последнего результата выполнена.\nВаш запрос:\n`{clean_prompt[:700]}`\n\n"
                    f"Применен прямой prompt без улучшения:\n`{clean_prompt[:700]}`"
                ),
                files=files,
            )
        except asyncio.TimeoutError:
            await interaction.followup.send("❌ Таймаут доработки изображения.")
        except Exception as exc:
            logger.exception(f"/refine_last failed: {exc}")
            await interaction.followup.send(f"❌ Ошибка refine_last: {exc}")

    async def run_prompt_enhancement(
        self,
        user_prompt: str,
        image_bytes: Optional[bytes],
        style_preset: Optional[str],
    ) -> PromptEnhancementResult:
        async def _task():
            return await self.prompt_enhancer.enhance_prompt(
                user_prompt=user_prompt,
                image_bytes=image_bytes,
                style_preset=style_preset,
            )

        return await self._run_in_queue(_task)

    async def run_image_generation(
        self,
        prompt: str,
        model: Optional[str] = None,
        image_count: Optional[int] = None,
    ) -> List[GeneratedImage]:
        async def _task():
            return await self.image_generator.generate_images(
                prompt=self._build_generation_prompt(prompt),
                image_count=image_count if image_count is not None else self.default_image_count,
                model=model,
            )

        return await self._run_in_queue(_task)

    async def run_image_edit(self, prompt: str, image_bytes: bytes) -> List[GeneratedImage]:
        async def _task():
            return await self.image_generator.edit_image(
                image_bytes=image_bytes,
                prompt=prompt,
                image_count=self.default_edit_image_count,
            )

        return await self._run_in_queue(_task)

    def _build_edit_prompt(self, user_prompt: str) -> str:
        return (
            f"{user_prompt.strip()}\n\n"
            "Strict edit rules:\n"
            "- Keep the original image identity and preserve all fine details (textures, micro details, edges, lighting nuances).\n"
            "- Keep composition, camera framing, geometry, proportions, and object layout unchanged unless explicitly requested.\n"
            "- Ensure all key subjects stay fully inside the frame with safe margins; avoid edge clipping or cut-off objects.\n"
            "- Apply only the requested edits and leave everything else intact.\n"
            "- If the request is about colors/palette, change only colors while preserving original materials and textures.\n"
            "- Do not add text, logos, watermarks, or new objects unless explicitly requested."
        )

    def _build_generation_prompt(self, user_prompt: str) -> str:
        return (
            f"{user_prompt.strip()}\n\n"
            "Framing rules:\n"
            "- Keep the main subject fully visible in frame.\n"
            "- Leave safe margins around key objects.\n"
            "- Avoid any cropping or cut-off objects unless explicitly requested."
        )

    def _build_skill_icon_prompt(self, user_prompt: str) -> str:
        return (
            f"{user_prompt.strip()}\n\n"
            "Skill icon rules:\n"
            "- Generate a game-style skill icon with bold, readable silhouette.\n"
            "- The skill art must fill the entire frame; avoid tiny centered objects.\n"
            "- Subject must touch or nearly touch all 4 edges of the frame.\n"
            "- No empty margins, no border, no vignette around the subject.\n"
            "- Do not leave visible background; the skill effect should occupy the full canvas.\n"
            "- Background must be solid black.\n"
            "- No transparency, no alpha holes, fully opaque render.\n"
            "- High contrast and clean edges suitable for small icon usage.\n"
            "- No text, letters, logos, or watermarks."
        )

    def _convert_images_to_skill_icons(self, images: List[GeneratedImage]) -> List[GeneratedImage]:
        converted: List[GeneratedImage] = []
        for image in images:
            converted.append(
                GeneratedImage(
                    image_bytes=self._convert_to_100x100_black_background(image.image_bytes),
                    revised_prompt=image.revised_prompt,
                )
            )
        return converted

    def _convert_to_100x100_black_background(self, image_bytes: bytes) -> bytes:
        with Image.open(io.BytesIO(image_bytes)) as source:
            rgba = source.convert("RGBA")
            content_bbox = self._detect_icon_content_bbox(rgba)
            if content_bbox:
                rgba = rgba.crop(content_bbox)

            composed = Image.new("RGB", rgba.size, (0, 0, 0))
            composed.paste(rgba, mask=rgba.getchannel("A"))
            composed = self._resize_cover(composed, (100, 100))
            output = io.BytesIO()
            composed.save(output, format="PNG")
            return output.getvalue()

    def _detect_icon_content_bbox(self, rgba: Image.Image) -> Optional[Tuple[int, int, int, int]]:
        width, height = rgba.size
        min_w = max(8, int(width * 0.2))
        min_h = max(8, int(height * 0.2))

        alpha_mask = rgba.getchannel("A").point(lambda value: 255 if value > 20 else 0)
        alpha_bbox = alpha_mask.getbbox()
        if self._is_valid_bbox(alpha_bbox, min_w=min_w, min_h=min_h):
            return self._expand_bbox(alpha_bbox, width=width, height=height, pad_ratio=0.03)

        rgb_mask = rgba.convert("RGB").point(lambda value: 255 if value > 18 else 0)
        rgb_bbox = rgb_mask.getbbox()
        if self._is_valid_bbox(rgb_bbox, min_w=min_w, min_h=min_h):
            return self._expand_bbox(rgb_bbox, width=width, height=height, pad_ratio=0.02)

        return None

    def _is_valid_bbox(
        self,
        bbox: Optional[Tuple[int, int, int, int]],
        min_w: int,
        min_h: int,
    ) -> bool:
        if not bbox:
            return False
        left, top, right, bottom = bbox
        return (right - left) >= min_w and (bottom - top) >= min_h

    def _expand_bbox(
        self,
        bbox: Tuple[int, int, int, int],
        width: int,
        height: int,
        pad_ratio: float,
    ) -> Tuple[int, int, int, int]:
        left, top, right, bottom = bbox
        pad_x = max(1, int((right - left) * pad_ratio))
        pad_y = max(1, int((bottom - top) * pad_ratio))
        return (
            max(0, left - pad_x),
            max(0, top - pad_y),
            min(width, right + pad_x),
            min(height, bottom + pad_y),
        )

    def _resize_cover(self, image: Image.Image, target_size: Tuple[int, int]) -> Image.Image:
        target_w, target_h = target_size
        src_w, src_h = image.size
        if src_w == 0 or src_h == 0:
            return image.resize(target_size)

        scale = max(target_w / src_w, target_h / src_h)
        resized_w = max(1, int(round(src_w * scale)))
        resized_h = max(1, int(round(src_h * scale)))
        if hasattr(Image, "Resampling"):
            resized = image.resize((resized_w, resized_h), Image.Resampling.LANCZOS)
        else:
            resized = image.resize((resized_w, resized_h), Image.LANCZOS)

        left = max(0, (resized_w - target_w) // 2)
        top = max(0, (resized_h - target_h) // 2)
        return resized.crop((left, top, left + target_w, top + target_h))

    def images_to_discord_files(self, images: List[GeneratedImage], filename_prefix: str = "imagine") -> List[discord.File]:
        files: List[discord.File] = []
        for index, image in enumerate(images, start=1):
            filename = f"{filename_prefix}_{index}.png"
            files.append(discord.File(io.BytesIO(image.image_bytes), filename=filename))
        return files

    def _remember_last_user_image(self, user_id: int, images: List[GeneratedImage]) -> None:
        if images:
            self._last_image_by_user[user_id] = images[0].image_bytes

    def _ensure_workers(self) -> None:
        if self._workers:
            return
        for _ in range(max(1, self._worker_count)):
            self._workers.append(asyncio.create_task(self._worker_loop()))

    async def _run_in_queue(self, task_coro_factory):
        self._ensure_workers()
        loop = asyncio.get_running_loop()
        result_future = loop.create_future()
        await self._task_queue.put((task_coro_factory, result_future))
        return await result_future

    async def _worker_loop(self):
        while True:
            task_coro_factory, result_future = await self._task_queue.get()
            try:
                result = await asyncio.wait_for(task_coro_factory(), timeout=self.task_timeout_seconds)
                if not result_future.done():
                    result_future.set_result(result)
            except Exception as exc:
                if not result_future.done():
                    result_future.set_exception(exc)
            finally:
                self._task_queue.task_done()
