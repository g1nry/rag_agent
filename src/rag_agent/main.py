from fastapi import FastAPI
from contextlib import asynccontextmanager
from pydantic import BaseModel
import logging

from .agents import red_team_agent
from .services.retrieval_service import retrieval_service
from .security.hitl import hitl_manager

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


# === НОВЫЙ ЭНДПОИНТ ===
@app.post("/api/agent/chat")
async def agent_chat(request: ChatRequest):
    result = await red_team_agent.ainvoke(
        message=request.message,
        thread_id=request.thread_id
    )
    return result


# === СТАРЫЙ ЭНДПОИНТ (для совместимости) ===
@app.post("/api/v1/chat")
async def legacy_chat(request: ChatRequest):
    """Старый эндпоинт — перенаправляет на нового агента"""
    result = await red_team_agent.ainvoke(
        message=request.message,
        thread_id=request.thread_id
    )
    return {
        "answer": result.get("response", ""),
        "contexts": []  # можно позже добавить реальные контексты
    }


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)