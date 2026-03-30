import base64
import json
import os
from dataclasses import dataclass
from typing import List, Optional

from openai import AsyncOpenAI


STYLE_PRESETS = {
    "ancient": "Apply an ancient handcrafted look with weathered materials and subtle historical wear.",
    "fantasy": "Apply a high-fantasy art direction with magical motifs, mystical lighting, and rich ornament.",
    "realistic": "Keep a realistic look with physically plausible materials, lighting, and fine texture detail.",
}


@dataclass
class PromptEnhancementResult:
    final_prompt: str
    variations: List[str]
    model: str


class PromptEnhancer:
    def __init__(self, api_key: str, model: Optional[str] = None, timeout_seconds: int = 35):
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model or os.getenv("PROMPT_ENHANCER_MODEL", "gpt-4.1")
        self.timeout_seconds = timeout_seconds

    async def enhance_prompt(
        self,
        user_prompt: str,
        image_bytes: Optional[bytes] = None,
        style_preset: Optional[str] = None,
    ) -> PromptEnhancementResult:
        image_included = image_bytes is not None
        preset_instruction = STYLE_PRESETS.get((style_preset or "").lower())
        payload = self._build_user_payload(user_prompt=user_prompt, image_included=image_included, preset_instruction=preset_instruction)

        content = [{"type": "text", "text": payload}]
        if image_bytes:
            image_data_url = self._to_data_url(image_bytes)
            content.append({"type": "image_url", "image_url": {"url": image_data_url}})

        response = await self.client.chat.completions.create(
            model=self.model,
            temperature=0.35,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": self._system_prompt()},
                {"role": "user", "content": content},
            ],
        )

        raw_json = response.choices[0].message.content or "{}"
        parsed = json.loads(raw_json)
        final_prompt = (parsed.get("final_prompt") or user_prompt).strip()
        raw_variations = parsed.get("variations") or []

        variations: List[str] = []
        for item in raw_variations:
            if isinstance(item, str) and item.strip():
                variations.append(item.strip())

        if not variations:
            variations = [final_prompt]
        variations = variations[:4]
        if len(variations) < 2:
            variations.append(f"{final_prompt}. Keep composition but alter style details.")

        return PromptEnhancementResult(
            final_prompt=final_prompt,
            variations=variations,
            model=self.model,
        )

    def _build_user_payload(self, user_prompt: str, image_included: bool, preset_instruction: Optional[str]) -> str:
        parts = [f"User request: {user_prompt.strip()}"]
        if image_included:
            parts.append(
                "Image is provided. You must produce prompts that are based on the provided image, "
                "preserve composition, and modify only specified elements."
            )
        if preset_instruction:
            parts.append(f"Style preset instruction: {preset_instruction}")
        return "\n".join(parts)

    def _system_prompt(self) -> str:
        return (
            "You are a prompt enhancer for image generation. "
            "Rewrite the user's request into one high quality production prompt and 2-4 alternatives. "
            "Return ONLY JSON with fields: final_prompt (string), variations (array of strings). "
            "If image is attached, preserve scene composition, structure, and layout while changing only requested elements. "
            "Use concrete visual details: materials, lighting, camera framing, color palette, texture, and style."
        )

    def _to_data_url(self, image_bytes: bytes) -> str:
        encoded = base64.b64encode(image_bytes).decode("ascii")
        return f"data:image/png;base64,{encoded}"
