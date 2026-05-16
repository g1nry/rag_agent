from fastapi import FastAPI
from contextlib import asynccontextmanager
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
import logging

from .agents import red_team_agent
from .services.retrieval_service import retrieval_service

logger = logging.getLogger(__name__)


class ChatRequest(BaseModel):
    message: str
    thread_id: str = "default"
    use_rag: bool = True
    top_k: int = 4


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Starting RedTeamAgent...")
    await red_team_agent.initialize(retrieval_service=retrieval_service)
    logger.info("✅ RedTeamAgent initialized")
    yield


app = FastAPI(title="RAG RedTeam Agent", version="0.2.0", lifespan=lifespan)


# === ГЛАВНАЯ СТРАНИЦА (для вебчика) ===
@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <html>
        <head><title>RAG RedTeam Agent</title></head>
        <body>
            <h1>🤖 RAG RedTeam Agent</h1>
            <p>Backend работает. Используй <code>POST /api/agent/chat</code> или <code>POST /api/v1/chat</code></p>
        </body>
    </html>
    """


# === ОСНОВНОЙ ЭНДПОИНТ ДЛЯ АГЕНТА ===
@app.post("/api/agent/chat")
async def agent_chat(request: ChatRequest):
    result = await red_team_agent.ainvoke(
        message=request.message,
        thread_id=request.thread_id
    )
    return {
        "answer": result.get("response", ""),
        "contexts": []
    }


# === СТАРЫЙ ЭНДПОИНТ (теперь тоже через агента) ===
@app.post("/api/v1/chat")
async def legacy_chat(request: ChatRequest):
    """Старый эндпоинт — теперь тоже использует RedTeamAgent"""
    result = await red_team_agent.ainvoke(
        message=request.message,
        thread_id=request.thread_id
    )
    return {
        "answer": result.get("response", ""),
        "contexts": []
    }


@app.get("/health")
async def health():
    return {"status": "ok", "agent_initialized": red_team_agent.initialized}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)