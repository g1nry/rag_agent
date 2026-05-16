from fastapi import FastAPI
from contextlib import asynccontextmanager
import logging

from .core.config import get_config
from .agents import red_team_agent
from .services.retrieval_service import retrieval_service  # твой существующий сервис

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("🚀 Starting RedTeamAgent...")
    await red_team_agent.initialize(retrieval_service=retrieval_service)
    logger.info("✅ RedTeamAgent initialized successfully")
    yield
    # Shutdown
    logger.info("🛑 Shutting down...")

app = FastAPI(
    title="RAG RedTeam Agent",
    description="Agentic RAG with LangGraph for LLM security research",
    version="0.2.0",
    lifespan=lifespan
)

# Существующие роуты оставляем как есть...
# Добавляем новый роут для агента

@app.post("/api/agent/chat")
async def agent_chat(request: dict):  # Можно сделать Pydantic модель позже
    message = request.get("message")
    thread_id = request.get("thread_id", "default")
    
    if not message:
        return {"error": "Message is required"}
    
    result = await red_team_agent.ainvoke(message=message, thread_id=thread_id)
    return result


# Для совместимости со старым кодом
@app.post("/chat")
async def legacy_chat(request: dict):
    """Legacy endpoint — можно перенаправлять на новый агент"""
    return await agent_chat(request)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)