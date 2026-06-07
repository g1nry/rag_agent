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


app = FastAPI(
    title="RAG RedTeam Agent",
    description="Agentic RAG + Dangerous Tools for LLM Security Research",
    version="0.3.0",
    lifespan=lifespan
)


@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <html>
        <head><title>RAG RedTeam Agent</title></head>
        <body style="font-family: sans-serif; padding: 40px;">
            <h1>🤖 RAG RedTeam Agent</h1>
            <p>Сервис запущен успешно.</p>
        </body>
    </html>
    """


# ==================== ПРОСТОЙ RAG (без tools) ====================
@app.post("/api/v1/chat")
async def simple_rag_chat(request: ChatRequest):
    """Простой RAG-chain: retrieve → prompt with context → LLM answer"""
    if not request.use_rag or not retrieval_service:
        # Fallback без RAG
        result = await red_team_agent.llm.ainvoke(request.message) if hasattr(red_team_agent, 'llm') else {"content": "RAG отключен"}
        return {"answer": getattr(result, 'content', str(result)), "contexts": []}

    # Получаем контексты
    retrieval_result = await retrieval_service.retrieve_context(
        request.message, 
        top_k=request.top_k
    )
    
    contexts = retrieval_result.get("contexts", []) if isinstance(retrieval_result, dict) else []
    
    if not contexts:
        answer = "Не удалось найти релевантную информацию в документах."
    else:
        # Формируем промпт с контекстом
        context_text = "\n\n".join([f"Документ {i+1}:\n{c.get('content', c.get('text', ''))}" 
                                  for i, c in enumerate(contexts)])
        
        prompt = f"""Используй только предоставленный контекст для ответа.
Отвечай на русском языке.

Контекст:
{context_text}

Вопрос: {request.message}

Ответ:"""

        # Вызываем LLM напрямую
        llm_result = await red_team_agent.llm.ainvoke(prompt)
        answer = getattr(llm_result, 'content', str(llm_result))

    return {
        "answer": answer,
        "contexts": contexts
    }


# ==================== ТОЛЬКО RETRIEVAL ====================
@app.post("/api/v1/rag/search")
async def rag_search(request: ChatRequest):
    """Только поиск по документам"""
    if not retrieval_service:
        return {"contexts": []}
    
    result = await retrieval_service.retrieve_context(request.message, top_k=request.top_k)
    contexts = result.get("contexts", []) if isinstance(result, dict) else []
    
    return {"contexts": contexts}


# ==================== ReAct АГЕНТ ====================
@app.post("/api/agent/chat")
async def agent_chat(request: ChatRequest):
    """Полноценный ReAct-агент с доступом ко всем инструментам"""
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
    return {
        "status": "ok",
        "agent_initialized": red_team_agent.initialized,
        "retrieval_service": retrieval_service is not None
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
