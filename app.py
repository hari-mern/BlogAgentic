import os

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request

from src.config import SUPPORTED_LANGUAGES
from src.graphs.graph_builder import GraphBuilder
from src.llms.groqllm import GroqLLM

load_dotenv()
os.environ.setdefault("LANGSMITH_API_KEY", os.getenv("LANGCHAIN_API_KEY", ""))

app = FastAPI(title="BlogAgentic", version="1.0.0")


@app.post("/blogs")
async def create_blogs(request: Request):
    data = await request.json()
    topic = (data.get("topic") or "").strip()
    language = (data.get("language") or "").strip().lower()

    if not topic:
        raise HTTPException(status_code=400, detail="'topic' is required.")

    if language and language not in SUPPORTED_LANGUAGES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported language '{language}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_LANGUAGES))}.",
        )

    llm = GroqLLM().get_llm()
    graph_builder = GraphBuilder(llm)

    if language:
        graph = graph_builder.setup_graph(usecase="language")
        state = graph.invoke({"topic": topic, "current_language": language})
    else:
        graph = graph_builder.setup_graph(usecase="topic")
        state = graph.invoke({"topic": topic})

    return {"data": state}


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)