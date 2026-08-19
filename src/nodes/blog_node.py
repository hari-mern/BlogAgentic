import re
from datetime import datetime

from langchain_core.messages import HumanMessage

from src.config import LLM_MAX_TOKENS, MAX_BLOG_CHARS, SUPPORTED_LANGUAGES
from src.states.blogstate import Blog, BlogState

# Matches reasoning/thinking blocks so the output works with both
# thinking and non-thinking models. The block always appears at the
# start of the raw model response.
_THINK_BLOCK = re.compile(
    r"^\s*<thinking>.*?</thinking>\s*|^\s*thinking\s*.*?\s*response\s*",
    flags=re.DOTALL | re.IGNORECASE,
)


def clean_response(content):
    """Strip reasoning/thinking blocks from a raw model response."""
    if not isinstance(content, str):
        return content
    return _THINK_BLOCK.sub("", content).strip()


def _blog_value(state, key):
    """Read a field off ``state["blog"]`` whether it is a dict or a model."""
    blog = state.get("blog") or {}
    if isinstance(blog, dict):
        return blog.get(key)
    return getattr(blog, key, None)


class BlogNode:
    """
    Node implementations for the blog generation graphs.
    """

    def __init__(self, llm):
        self.llm = llm

    def title_creation(self, state: BlogState):
        topic = (state.get("topic") or "").strip()
        if not topic:
            raise ValueError("A topic is required to create a blog title.")

        prompt = (
            "you are an expert blog content writer. use markdown formatting.\n"
            'generate a blog title for the topic "{topic}". this title should be '
            "creative and seo friendly. use the current year {year} when needed.\n"
            "return only the title as a single markdown heading. do not include any "
            "explanation, bullet points, or extra text."
        )
        response = self.llm.invoke(
            prompt.format(topic=topic, year=datetime.now().year)
        )
        title = clean_response(response.content).lstrip("#").strip()
        if not title:
            raise ValueError("The model returned an empty blog title.")
        return {"blog": {"title": title}}

    def content_generator(self, state: BlogState):
        topic = (state.get("topic") or "").strip()
        title = _blog_value(state, "title")
        if not topic or not title:
            raise ValueError("Both 'topic' and a blog title are required to generate content.")

        system_prompt = (
            "you are an expert blog writer. use markdown formatting.\n"
            'generate a detailed blog content for the topic "{topic}" in the year {year}.\n'
            'start the blog with the markdown heading "# {title}" and then write the '
            "content with a clear breakdown of sections.\n"
            "keep the blog concise: between 1500 and {max_chars} characters."
        )
        response = self.llm.invoke(
            system_prompt.format(
                topic=topic, title=title, year=datetime.now().year, max_chars=MAX_BLOG_CHARS
            )
        )
        content = clean_response(response.content)

        # Hard safeguard: never exceed the configured budget so the downstream
        # translation request stays inside the token/TPM limits.
        if len(content) > MAX_BLOG_CHARS:
            content = content[:MAX_BLOG_CHARS].rstrip()
        if not content:
            raise ValueError("The model returned empty blog content.")
        return {"blog": {"title": title, "content": content}}

    def translation(self, state: BlogState):
        language = state["current_language"].lower()
        blog_title = _blog_value(state, "title")
        blog_content = _blog_value(state, "content") or ""
        blog_content = blog_content[:MAX_BLOG_CHARS]

        translation_prompt = (
            "Translate the following content into {language}.\n"
            "- maintain the original tone, style, and formatting.\n"
            "- adapt cultural references and idioms to be appropriate for {language}.\n"
            "- keep the title exactly as: {blog_title}\n"
            '- return ONLY a JSON object with keys "title" and "content". do not '
            "include any explanation.\n\n"
            "ORIGINAL CONTENT:\n{blog_content}"
        )
        messages = [
            HumanMessage(
                translation_prompt.format(
                    language=language, blog_title=blog_title, blog_content=blog_content
                )
            )
        ]

        try:
            result = self.llm.with_structured_output(
                Blog, method="json_mode", max_tokens=LLM_MAX_TOKENS
            ).invoke(messages)
        except Exception as exc:
            raise RuntimeError(f"Translation to '{language}' failed: {exc}") from exc

        if not result or not result.content:
            raise RuntimeError(f"Translation to '{language}' returned an empty result.")
        return {"blog": {"title": result.title, "content": result.content}}

    def route(self, state: BlogState):
        return {"current_language": state["current_language"].lower()}

    def route_decision(self, state: BlogState):
        language = state["current_language"].lower()
        if language in SUPPORTED_LANGUAGES:
            return language
        raise ValueError(
            f"Unsupported language '{language}'. "
            f"Supported languages: {', '.join(sorted(SUPPORTED_LANGUAGES))}."
        )