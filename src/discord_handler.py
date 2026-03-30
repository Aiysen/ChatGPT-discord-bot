import asyncio
import io
import os
import time
from collections import defaultdict, deque
from typing import List, Optional

import discord
from discord import app_commands

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
    ):
        self.workflow = workflow
        self.enhancement = enhancement
        self.original_prompt = original_prompt
        self.owner_id = owner_id
        self.had_input_image = had_input_image

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
            images = await self.workflow.run_image_generation(selected_prompt)
            files = self.workflow.images_to_discord_files(images)

            self.workflow.history_store.save(
                GenerationRecord(
                    user_id=interaction.user.id,
                    username=str(interaction.user),
                    original_prompt=self.original_prompt,
                    final_prompt=self.enhancement.final_prompt,
                    variations=self.enhancement.variations,
                    chosen_prompt=selected_prompt,
                    prompt_model=self.enhancement.model,
                    image_model=self.workflow.image_generator.model,
                    image_count=len(images),
                    had_input_image=self.had_input_image,
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
    ):
        super().__init__(timeout=600)
        self.add_item(
            VariationSelect(
                workflow=workflow,
                enhancement=enhancement,
                original_prompt=original_prompt,
                owner_id=owner_id,
                had_input_image=had_input_image,
            )
        )


class DiscordImageHandler:
    def __init__(self, discord_client: discord.Client):
        api_key = os.getenv("OPENAI_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_KEY is required for /imagine and /variations commands")

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
            images = await self.run_image_generation(clean_prompt)
            files = self.images_to_discord_files(images)

            self.history_store.save(
                GenerationRecord(
                    user_id=interaction.user.id,
                    username=str(interaction.user),
                    original_prompt=clean_prompt,
                    final_prompt=clean_prompt,
                    variations=[clean_prompt],
                    chosen_prompt=clean_prompt,
                    prompt_model="none",
                    image_model=self.image_generator.model,
                    image_count=len(images),
                    had_input_image=False,
                )
            )

            await interaction.followup.send(
                content=(
                    f"Сгенерировано без улучшения промта:\n`{clean_prompt[:700]}`\n\n"
                    f"Модель генерации: `{self.image_generator.model}`"
                ),
                files=files,
            )
        except asyncio.TimeoutError:
            await interaction.followup.send("❌ Таймаут обработки запроса. Попробуйте снова.")
        except Exception as exc:
            logger.exception(f"/draw failed: {exc}")
            await interaction.followup.send(f"❌ Ошибка генерации: {exc}")

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
            images = await self.run_image_edit(clean_prompt, image_bytes)
            files = self.images_to_discord_files(images)

            self.history_store.save(
                GenerationRecord(
                    user_id=interaction.user.id,
                    username=str(interaction.user),
                    original_prompt=clean_prompt,
                    final_prompt=clean_prompt,
                    variations=[clean_prompt],
                    chosen_prompt=clean_prompt,
                    prompt_model="none",
                    image_model=self.image_generator.edit_model,
                    image_count=len(images),
                    had_input_image=True,
                )
            )

            await interaction.followup.send(
                content=(
                    f"Изображение отредактировано по вашему prompt:\n`{clean_prompt[:700]}`\n\n"
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
            request_text = (
                "Create 4 meaningful prompt variations for this base prompt. "
                "Keep subject and composition close, but vary rendering approach, materials and lighting:\n"
                f"{record.chosen_prompt}"
            )
            enhancement = await self.run_prompt_enhancement(user_prompt=request_text, image_bytes=None, style_preset=None)
            selected_prompt = enhancement.variations[0]
            images = await self.run_image_generation(selected_prompt)
            files = self.images_to_discord_files(images)

            self.history_store.save(
                GenerationRecord(
                    user_id=interaction.user.id,
                    username=str(interaction.user),
                    original_prompt=record.original_prompt,
                    final_prompt=record.final_prompt,
                    variations=enhancement.variations,
                    chosen_prompt=selected_prompt,
                    prompt_model=enhancement.model,
                    image_model=self.image_generator.model,
                    image_count=len(images),
                    had_input_image=record.had_input_image,
                )
            )

            view = VariationView(
                workflow=self,
                enhancement=enhancement,
                original_prompt=record.original_prompt,
                owner_id=interaction.user.id,
                had_input_image=record.had_input_image,
            )
            await interaction.followup.send(
                content=f"Сгенерированы новые вариации на базе последнего результата.\nБазовый вариант:\n`{selected_prompt[:700]}`",
                files=files,
                view=view,
            )
        except asyncio.TimeoutError:
            await interaction.followup.send("❌ Таймаут генерации вариаций.")
        except Exception as exc:
            logger.exception(f"/variations failed: {exc}")
            await interaction.followup.send(f"❌ Ошибка вариаций: {exc}")

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

    async def run_image_generation(self, prompt: str) -> List[GeneratedImage]:
        async def _task():
            return await self.image_generator.generate_images(
                prompt=prompt,
                image_count=self.default_image_count,
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

    def images_to_discord_files(self, images: List[GeneratedImage]) -> List[discord.File]:
        files: List[discord.File] = []
        for index, image in enumerate(images, start=1):
            filename = f"imagine_{index}.png"
            files.append(discord.File(io.BytesIO(image.image_bytes), filename=filename))
        return files

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
