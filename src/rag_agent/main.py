from fastapi import FastAPI, HTTPException, UploadFile, File
from contextlib import asynccontextmanager
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
import logging

from .agents import red_team_agent
from .services.retrieval_service import retrieval_service
from .services.document_ingestion_service import DocumentIngestionService
from .storage.vector_store import VectorStore

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

# Создаём экземпляр сервиса загрузки
ingestion_service = DocumentIngestionService()


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


# ==================== ЗАГРУЗКА ДОКУМЕНТОВ ====================
@app.post("/api/v1/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    """Загрузка и индексация документа"""
    try:
        content = await file.read()
        result = await ingestion_service.ingest_document(
            filename=file.filename,
            content=content
        )
        return result
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=400, detail=str(e))


# ==================== УПРАВЛЕНИЕ ДОКУМЕНТАМИ ====================
@app.get("/api/v1/documents")
async def list_documents():
    try:
        docs = red_team_agent.vector_store.get_documents() if hasattr(red_team_agent, 'vector_store') else []
        return {"documents": docs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/v1/documents/{filename}")
async def delete_document(filename: str):
    try:
        success = red_team_agent.vector_store.delete_document(filename) if hasattr(red_team_agent, 'vector_store') else False
        if success:
            return {"status": "deleted", "filename": filename}
        raise HTTPException(status_code=404, detail="Документ не найден")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== RAG ====================
@app.post("/api/v1/chat")
async def simple_rag_chat(request: ChatRequest):
    if not request.use_rag or not retrieval_service:
        result = await red_team_agent.llm.ainvoke(request.message)
        return {"answer": result.content, "contexts": []}

    retrieval_result = await retrieval_service.retrieve_context(request.message, top_k=request.top_k)
    contexts = retrieval_result.get("contexts", []) if isinstance(retrieval_result, dict) else []

    if not contexts:
        return {"answer": "Релевантная информация не найдена.", "contexts": []}

    context_text = "\n\n".join([f"[{i+1}] {c.get('text', '')}" for i, c in enumerate(contexts)])
    prompt = f"Используй только предоставленный контекст.\n\n{context_text}\n\nВопрос: {request.message}\n\nОтвет:"

    llm_result = await red_team_agent.llm.ainvoke(prompt)
    return {"answer": llm_result.content, "contexts": contexts}


@app.post("/api/v1/rag/search")
async def rag_search(request: ChatRequest):
    if not retrieval_service:
        return {"contexts": []}
    result = await retrieval_service.retrieve_context(request.message, top_k=request.top_k)
    return {"contexts": result.get("contexts", []) if isinstance(result, dict) else []}


# ==================== ReAct АГЕНТ ====================
@app.post("/api/agent/chat")
async def agent_chat(request: ChatRequest):
    result = await red_team_agent.ainvoke(message=request.message, thread_id=request.thread_id)
    return {"answer": result.get("response", ""), "contexts": []}


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
