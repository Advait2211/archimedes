import os
import logging
from pathlib import Path

import openai
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

logger = logging.getLogger(__name__)


class NIMClient:
    TEXT_MODEL = "meta/llama-3.3-70b-instruct"
    VISION_MODEL = "meta/llama-3.2-90b-vision-instruct"
    BASE_URL = "https://integrate.api.nvidia.com/v1"

    def __init__(self, api_key: str = None):
        key = api_key or os.environ.get("NVIDIA_API_KEY")
        if not key:
            raise ValueError(
                "NVIDIA API key required. Set NVIDIA_API_KEY environment variable "
                "or pass api_key to NIMClient."
            )
        self.client = openai.OpenAI(base_url=self.BASE_URL, api_key=key)

    def complete_text(self, system: str, user: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.TEXT_MODEL,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                max_tokens=8192,
                temperature=0.7,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"NIM text completion failed: {e}")
            raise

    def complete_vision(
        self, system: str, user_text: str, image_b64_list: list[str]
    ) -> str:
        content = [{"type": "text", "text": user_text}]
        for b64 in image_b64_list:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
            })

        try:
            response = self.client.chat.completions.create(
                model=self.VISION_MODEL,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": content},
                ],
                max_tokens=8192,
                temperature=0.5,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"NIM vision completion failed: {e}")
            raise
