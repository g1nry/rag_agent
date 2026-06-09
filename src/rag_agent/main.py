from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks, Depends
from contextlib import asynccontextmanager
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
import logging
import uuid

from .agents import red_team_agent
from .core.config import get_settings
from .integrations.ollama.client import OllamaClient
from .services.llm_service import LLMService
from .services.retrieval_service import retrieval_service
from .services.document_ingestion_service import DocumentIngestionService
from .storage.document_store import DocumentStore
from .storage.vector_store import JsonVectorStore, VectorStore

logger = logging.getLogger(__name__)

_settings = get_settings()
_ollama_client = OllamaClient(str(_settings.ollama_base_url))
_llm_service = LLMService(client=_ollama_client, settings=_settings)
_document_store = DocumentStore(_settings.documents_dir)
_vector_store = JsonVectorStore(_settings.index_path)

# Создаём экземпляр сервиса загрузки
ingestion_service = DocumentIngestionService(
    settings=_settings,
    document_store=_document_store,
    vector_store=_vector_store,
    llm_service=_llm_service,
)

document_status_store: dict[str, dict] = {}

def get_document_ingestion_service() -> DocumentIngestionService:
    return ingestion_service

async def _process_document_ingestion(
    document_id: str,
    filename: str,
    content: bytes,
    service: DocumentIngestionService,
):
    document_status_store[document_id] = {
        "document_id": document_id,
        "filename": filename,
        "status": "indexing",
        "chunks_indexed": None,
        "error": None,
        "message": None,
    }
    try:
        result = await service.ingest_document(filename=filename, content=content)
        document_status_store[document_id].update(
            status="indexed",
            chunks_indexed=result.chunks_indexed,
        )
    except Exception as exc:
        document_status_store[document_id].update(
            status="failed",
            error=str(exc),
            message=getattr(exc, "detail", str(exc)),
        )


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


# ==================== ЗАГРУЗКА ДОКУМЕНТОВ ====================
@app.post("/api/v1/documents/upload")
@app.post("/chat/v1/documents/upload")
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    ingestion_service: DocumentIngestionService = Depends(get_document_ingestion_service),
):
    """Загрузка и индексация документа"""
    try:
        content = await file.read()
        document_id = str(uuid.uuid4())
        document_status_store[document_id] = {
            "document_id": document_id,
            "filename": file.filename,
            "status": "queued",
            "chunks_indexed": None,
            "error": None,
            "message": None,
        }
        background_tasks.add_task(
            _process_document_ingestion,
            document_id,
            file.filename,
            content,
            ingestion_service,
        )
        return {
            "document_id": document_id,
            "filename": file.filename,
            "status": "queued",
        }
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/v1/documents/{document_id}/status")
async def get_document_status(document_id: str):
    status = document_status_store.get(document_id)
    if not status:
        raise HTTPException(status_code=404, detail="Document not found")
    return status


# ==================== УПРАВЛЕНИЕ ДОКУМЕНТАМИ ====================
@app.get("/api/v1/documents")
async def list_documents():
    try:
        docs = retrieval_service._vector_store.get_documents() if retrieval_service and hasattr(retrieval_service, '_vector_store') else []
        return {"documents": docs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/v1/documents/{filename}")
async def delete_document(filename: str):
    try:
        success = retrieval_service._vector_store.delete_document(filename) if retrieval_service and hasattr(retrieval_service, '_vector_store') else False
        if success:
            return {"status": "deleted", "filename": filename}
        raise HTTPException(status_code=404, detail="Документ не найден")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== RAG ====================
@app.post("/api/v1/chat")
async def simple_rag_chat(request: ChatRequest):
    if not red_team_agent.initialized:
        await red_team_agent.initialize(retrieval_service=retrieval_service)

    if not request.use_rag or not retrieval_service:
        result = await red_team_agent.llm.ainvoke(request.message)
        return {"answer": result.content, "contexts": []}

    retrieval_result = await retrieval_service.retrieve_context(request.message, top_k=request.top_k)
    contexts = retrieval_result.contexts if hasattr(retrieval_result, "contexts") else []

    if not contexts:
        return {"answer": "Релевантная информация не найдена.", "contexts": []}

    context_text = "\n\n".join([f"[{i+1}] {c.get('text', '')}" for i, c in enumerate(contexts)])
    prompt = f"Используй только предоставленный контекст.\n\n{context_text}\n\nВопрос: {request.message}\n\nОтвет:"

    llm_result = await red_team_agent.llm.ainvoke(prompt)
    return {"answer": llm_result.content, "contexts": contexts}


@app.post("/api/v1/rag/search")
@app.post("/api/v1/rag/chat")
async def rag_search(request: ChatRequest):
    if not retrieval_service:
        return {"contexts": []}
    result = await retrieval_service.retrieve_context(request.message, top_k=request.top_k)
    return {"contexts": result.contexts if hasattr(result, "contexts") else []}


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
