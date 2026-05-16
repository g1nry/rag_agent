from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from pydantic import BaseModel
import logging

from .agents import red_team_agent
from .services.retrieval_service import retrieval_service  # твой существующий сервис
from .security.hitl import hitl_manager

logger = logging.getLogger(__name__)


class ChatRequest(BaseModel):
    message: str
    thread_id: str = "default"
    auto_confirm_medium: bool = True


class HITLConfirmRequest(BaseModel):
    tool_name: str
    args: dict
    confirmed: bool


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Starting RedTeamAgent...")
    await red_team_agent.initialize(retrieval_service=retrieval_service)
    
    # Пример callback для HITL (в реальном проекте можно заменить на WebSocket/очередь)
    async def confirmation_callback(tool_name: str, args: dict, risk_level: str) -> bool:
        logger.warning(f"⚠️  HITL required for {tool_name} (risk: {risk_level})")
        # В продакшене здесь будет запрос к фронтенду/пользователю
        return False  # По умолчанию не подтверждаем автоматически
    
    hitl_manager.set_callback(confirmation_callback)
    logger.info("✅ RedTeamAgent + HITL initialized")
    yield
    logger.info("🛑 Shutting down...")


app = FastAPI(
    title="RAG RedTeam Agent",
    description="Agentic RAG with LangGraph for LLM security research",
    version="0.2.0",
    lifespan=lifespan
)


@app.post("/api/agent/chat")
async def agent_chat(request: ChatRequest):
    """Основной endpoint для общения с агентом"""
    result = await red_team_agent.ainvoke(
        message=request.message,
        thread_id=request.thread_id,
        auto_confirm_medium=request.auto_confirm_medium
    )
    return result


@app.post("/api/agent/hitl/confirm")
async def hitl_confirm(request: HITLConfirmRequest):
    """Endpoint для подтверждения опасных действий"""
    # Здесь можно добавить логику сохранения подтверждения
    logger.info(f"HITL confirmation for {request.tool_name}: {request.confirmed}")
    return {"status": "ok", "confirmed": request.confirmed}


@app.get("/health")
async def health():
    return {"status": "ok", "agent_initialized": red_team_agent.initialized}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)