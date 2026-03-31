import base64
import inspect
import io
import os
from dataclasses import dataclass
from typing import List, Optional

import aiohttp
from openai import AsyncOpenAI
from PIL import Image


@dataclass
class GeneratedImage:
    image_bytes: bytes
    revised_prompt: Optional[str] = None


class ImageGenerator:
    def __init__(self, api_key: str, model: Optional[str] = None):
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model or os.getenv("IMAGE_GENERATION_MODEL", "gpt-image-1")
        self.edit_model = os.getenv("IMAGE_EDIT_MODEL", "gpt-image-1.5")
        self.generation_size = self._resolve_generate_size(os.getenv("IMAGE_GENERATION_SIZE", "auto"))
        self.edit_size = os.getenv("IMAGE_EDIT_SIZE", "match_input").strip().lower()

    async def generate_images(
        self,
        prompt: str,
        image_count: int = 1,
        size: Optional[str] = None,
        quality: str = "auto",
    ) -> List[GeneratedImage]:
        safe_count = max(1, min(image_count, 4))
        requested_size = size or self.generation_size
        generate_kwargs = {
            "model": self.model,
            "prompt": prompt,
            "n": safe_count,
            "quality": quality,
        }
        if requested_size != "auto":
            generate_kwargs["size"] = requested_size
        response = await self.client.images.generate(**generate_kwargs)

        result: List[GeneratedImage] = []
        for item in response.data:
            if getattr(item, "b64_json", None):
                result.append(
                    GeneratedImage(
                        image_bytes=base64.b64decode(item.b64_json),
                        revised_prompt=getattr(item, "revised_prompt", None),
                    )
                )
                continue

            if getattr(item, "url", None):
                downloaded = await self._download_image(item.url)
                result.append(GeneratedImage(image_bytes=downloaded))

        if not result:
            raise RuntimeError("Image API returned no images")
        return result

    async def edit_image(
        self,
        image_bytes: bytes,
        prompt: str,
        image_count: int = 1,
        size: Optional[str] = None,
        quality: str = "auto",
        background: str = "auto",
        moderation: str = "auto",
        input_fidelity: str = "high",
    ) -> List[GeneratedImage]:
        safe_count = 1
        requested_size = self._resolve_edit_size(size=size, image_bytes=image_bytes)
        prepared_image_bytes = self._prepare_image_for_edit(image_bytes)
        image_file = io.BytesIO(prepared_image_bytes)
        image_file.name = "input.png"

        optional_kwargs = {
            "quality": quality,
            "background": background,
            "moderation": moderation,
            "input_fidelity": input_fidelity,
        }
        edit_kwargs = self._filter_supported_image_edit_kwargs(optional_kwargs)

        edit_request_kwargs = {
            "model": self.edit_model,
            "image": image_file,
            "prompt": prompt,
            "n": safe_count,
            **edit_kwargs,
        }
        if requested_size != "auto":
            edit_request_kwargs["size"] = requested_size
        response = await self.client.images.edit(**edit_request_kwargs)

        result: List[GeneratedImage] = []
        for item in response.data:
            if getattr(item, "b64_json", None):
                result.append(
                    GeneratedImage(
                        image_bytes=base64.b64decode(item.b64_json),
                        revised_prompt=getattr(item, "revised_prompt", None),
                    )
                )
                continue

            if getattr(item, "url", None):
                downloaded = await self._download_image(item.url)
                result.append(GeneratedImage(image_bytes=downloaded))

        if not result:
            raise RuntimeError("Image edit API returned no images")
        return result

    async def _download_image(self, url: str) -> bytes:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status != 200:
                    raise RuntimeError(f"Failed to download image: HTTP {response.status}")
                return await response.read()

    def _prepare_image_for_edit(self, image_bytes: bytes) -> bytes:
        """Normalize image mode for OpenAI edit API (expects alpha or grayscale modes)."""
        with Image.open(io.BytesIO(image_bytes)) as source:
            if source.mode in {"RGBA", "LA", "L"} and source.format == "PNG":
                return image_bytes

            normalized = source.convert("RGBA")
            output = io.BytesIO()
            normalized.save(output, format="PNG")
            return output.getvalue()

    def _filter_supported_image_edit_kwargs(self, options: dict) -> dict:
        """Keep only kwargs supported by current images.edit SDK signature."""
        try:
            params = inspect.signature(self.client.images.edit).parameters
        except (TypeError, ValueError):
            return {}

        return {key: value for key, value in options.items() if key in params}

    def _resolve_generate_size(self, size: str) -> str:
        normalized = (size or "").strip().lower()
        if normalized in {"auto", "1024x1024", "1536x1024", "1024x1536"}:
            return normalized
        return "auto"

    def _resolve_edit_size(self, size: Optional[str], image_bytes: bytes) -> str:
        if size:
            return self._resolve_generate_size(size)
        if self.edit_size == "match_input":
            return self._size_from_input_aspect(image_bytes)
        return self._resolve_generate_size(self.edit_size)

    def _size_from_input_aspect(self, image_bytes: bytes) -> str:
        with Image.open(io.BytesIO(image_bytes)) as source:
            width, height = source.size
        if width > height:
            return "1536x1024"
        if height > width:
            return "1024x1536"
        return "1024x1024"
