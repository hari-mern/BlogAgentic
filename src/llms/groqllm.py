import os

from langchain_groq import ChatGroq

from src.config import GROQ_MODEL, GROQ_REASONING_EFFORT


class GroqLLM:
    """Factory that builds a configured Groq chat model."""

    def get_llm(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY is not set. Add it to the .env file.")

        try:
            llm_kwargs = {"api_key": api_key, "model_name": GROQ_MODEL}
            if "qwen" in GROQ_MODEL:
                llm_kwargs["reasoning_effort"] = GROQ_REASONING_EFFORT
            return ChatGroq(**llm_kwargs)
        except Exception as exc:
            raise ValueError(f"Failed to initialise Groq LLM: {exc}") from exc