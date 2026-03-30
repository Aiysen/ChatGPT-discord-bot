import base64
import os
from dataclasses import dataclass
from typing import List, Optional

import aiohttp
from openai import AsyncOpenAI


@dataclass
class GeneratedImage:
    image_bytes: bytes
    revised_prompt: Optional[str] = None


class ImageGenerator:
    def __init__(self, api_key: str, model: Optional[str] = None):
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model or os.getenv("IMAGE_GENERATION_MODEL", "gpt-image-1")

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

    async def _download_image(self, url: str) -> bytes:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status != 200:
                    raise RuntimeError(f"Failed to download image: HTTP {response.status}")
                return await response.read()
