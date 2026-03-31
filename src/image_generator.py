import base64
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

    async def generate_images(
        self,
        prompt: str,
        image_count: int = 1,
        size: str = "1024x1024",
        quality: str = "auto",
    ) -> List[GeneratedImage]:
        safe_count = max(1, min(image_count, 4))
        response = await self.client.images.generate(
            model=self.model,
            prompt=prompt,
            n=safe_count,
            size=size,
            quality=quality,
        )

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
        size: str = "1024x1024",
        quality: str = "auto",
        background: str = "auto",
        moderation: str = "auto",
        input_fidelity: str = "high",
    ) -> List[GeneratedImage]:
        safe_count = 1
        prepared_image_bytes = self._prepare_image_for_edit(image_bytes)
        image_file = io.BytesIO(prepared_image_bytes)
        image_file.name = "input.png"

        response = await self.client.images.edit(
            model=self.edit_model,
            image=image_file,
            prompt=prompt,
            n=safe_count,
            size=size,
            quality=quality,
            background=background,
            moderation=moderation,
            input_fidelity=input_fidelity,
        )

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
