"""
Centralised application configuration.

All tunable values come from environment variables with sensible defaults,
so the app behaves the same way locally, in LangGraph Studio, and in CI.
"""

import os

from dotenv import load_dotenv

load_dotenv()


def _get_int(name: str, default: int) -> int:
    """Read an integer env var, falling back to ``default`` on missing/invalid values."""
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


GROQ_MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3.6-27b")
GROQ_REASONING_EFFORT = os.getenv("GROQ_REASONING_EFFORT", "none")

# Maximum number of output tokens for a single model call.
LLM_MAX_TOKENS = _get_int("LLM_MAX_TOKENS", 8000)

# Maximum length of the generated blog content. This keeps every request
# (input + output) well inside the Groq free-tier TPM budget (8,000 tokens/min).
MAX_BLOG_CHARS = _get_int("MAX_BLOG_CHARS", 2500)

# Languages supported by the translation graph. Keys are the lowercase language
# names used by the API and the routing node.
SUPPORTED_LANGUAGES = {
    "hindi": "hindi",
    "french": "french",
}