from __future__ import annotations

import os

from dotenv import load_dotenv


load_dotenv()


def get_openai_settings() -> dict[str, str]:
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is missing. Add it to your environment or .env file.")

    return {
        "api_key": api_key,
        "base_url": os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        "chat_model": os.getenv("OPENAI_CHAT_MODEL", "gpt-4.1-mini"),
        "embedding_model": os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
    }
