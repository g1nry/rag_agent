from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import BackgroundTasks, Depends, FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
import logging

from .agents import red_team_agent
from .core.config import get_settings
from .domain.schemas import DocumentStatusResponse, DocumentUploadResponse
from .services.retrieval_service import retrieval_service
from .integrations.ollama.client import OllamaClient
from .guardrails.guardrails import check_message_guardrails, check_output_guardrails
from .services.document_errors import DocumentIngestionError
from .services.document_ingestion_service import DocumentIngestionService
from .services.llm_service import LLMService
from .storage.document_store import DocumentStore
from .storage.vector_store import JsonVectorStore

logger = logging.getLogger(__name__)
settings = get_settings()
rag_llm_service = LLMService(
    client=OllamaClient(str(settings.ollama_base_url)),
    settings=settings,
)
document_ingestion_service = DocumentIngestionService(
    settings=settings,
    document_store=DocumentStore(settings.documents_dir),
    vector_store=JsonVectorStore(settings.index_path),
    llm_service=rag_llm_service,
)
document_ingestion_jobs: dict[str, DocumentStatusResponse] = {}


class ChatRequest(BaseModel):
    message: str
    thread_id: str = "default"


class RAGChatRequest(ChatRequest):
    top_k: int = 4


def get_document_ingestion_service() -> DocumentIngestionService:
    return document_ingestion_service


def _set_document_status(status: DocumentStatusResponse) -> None:
    document_ingestion_jobs[status.document_id] = status


async def _run_document_ingestion_job(
    document_id: str,
    filename: str,
    content: bytes,
    ingestion_service: DocumentIngestionService,
) -> None:
    _set_document_status(
        DocumentStatusResponse(
            document_id=document_id,
            filename=filename,
            status="indexing",
        )
    )

    try:
        result = await ingestion_service.ingest_document(filename, content)
    except DocumentIngestionError as exc:
        _set_document_status(
            DocumentStatusResponse(
                document_id=document_id,
                filename=filename,
                status="failed",
                error=exc.error_code,
                message=exc.detail,
            )
        )
        return
    except Exception as exc:
        logger.exception("Document ingestion job failed")
        _set_document_status(
            DocumentStatusResponse(
                document_id=document_id,
                filename=filename,
                status="failed",
                error="document_ingestion_failed",
                message=str(exc),
            )
        )
        return

    _set_document_status(
        DocumentStatusResponse(
            document_id=document_id,
            filename=result.filename,
            status="indexed",
            chunks_indexed=result.chunks_indexed,
        )
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Starting RedTeamAgent...")
    await red_team_agent.initialize(retrieval_service=retrieval_service)
    logger.info("✅ RedTeamAgent initialized")
    yield


app = FastAPI(
    title="RAG RedTeam Agent",
    description="Agentic RAG + dangerous tools for LLM security research",
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
            <p>Backend is running.</p>
            <p>Main endpoints:</p>
            <ul>
                <li><code>POST /api/v1/chat</code></li>
                <li><code>POST /api/v1/rag/chat</code></li>
                <li><code>POST /api/agent/chat</code></li>
                <li><code>GET /health</code></li>
            </ul>
        </body>
    </html>
    """


async def _check_input_guardrail(message: str):
    # === INPUT RAIL ===
    blocked_in = await check_message_guardrails(
        [{"role": "user", "content": message}]
    )
    if blocked_in:
        logger.warning(f"Input rail blocked: {message[:80]}...")
        return {
            "answer": blocked_in,
            "contexts": [],
            "blocked": True,
            "blocked_by": "input_rail",
        }
    return None


async def _run_agent_message(
    *,
    message: str,
    thread_id: str,
    contexts: list | None = None,
):
    contexts = contexts or []

    result = await red_team_agent.ainvoke(
        message=message,
        thread_id=thread_id,
    )
    answer = result.get("response", "")

    # === OUTPUT RAIL (защита от RAG poisoning) ===
    blocked_out = await check_output_guardrails(answer)
    if blocked_out:
        logger.warning("Output rail blocked agent response")
        return {
            "answer": blocked_out,
            "contexts": [context.model_dump() for context in contexts],
            "blocked": True,
            "blocked_by": "output_rail",
        }

    return {
        "answer": answer,
        "contexts": [context.model_dump() for context in contexts],
    }


async def _guarded_chat(request: ChatRequest):
    blocked = await _check_input_guardrail(request.message)
    if blocked:
        return blocked

    return await _run_agent_message(
        message=request.message,
        thread_id=request.thread_id,
    )


async def _guarded_rag_chat(request: RAGChatRequest):
    blocked = await _check_input_guardrail(request.message)
    if blocked:
        return blocked

    retrieval_result = await retrieval_service.retrieve_context(
        query=request.message,
        top_k=request.top_k,
    )
    contexts = retrieval_result.contexts

    if not contexts:
        return {
            "answer": "Не удалось найти релевантный контекст в загруженных документах.",
            "contexts": [],
        }

    formatted_contexts = "\n\n".join(
        (
            f"[{index}] source={context.source}, chunk_id={context.chunk_id}\n"
            f"{context.text}"
        )
        for index, context in enumerate(contexts, start=1)
    )
    prompt = (
        "You are a RAG assistant. Answer in the same language as the user. "
        "Use only the RAG context below. If the context is not enough, say that directly. "
        "Do not call or mention tools, shell commands, file operations, or rag_search.\n\n"
        f"RAG context:\n{formatted_contexts}\n\n"
        f"User question:\n{request.message}"
    )
    answer = await rag_llm_service.generate(prompt)

    blocked_out = await check_output_guardrails(answer)
    if blocked_out:
        logger.warning("Output rail blocked RAG response")
        return {
            "answer": blocked_out,
            "contexts": [context.model_dump() for context in contexts],
            "blocked": True,
            "blocked_by": "output_rail",
        }

    return {
        "answer": answer,
        "contexts": [context.model_dump() for context in contexts],
    }


@app.post("/api/agent/chat")
async def agent_chat(request: ChatRequest):
    return await _guarded_chat(request)


@app.post("/api/v1/chat")
async def legacy_chat(request: ChatRequest):
    return await _guarded_chat(request)


@app.post("/api/v1/rag/chat")
async def rag_chat(request: RAGChatRequest):
    return await _guarded_rag_chat(request)


@app.post("/chat/v1/documents/upload", response_model=DocumentUploadResponse)
@app.post("/api/v1/documents/upload", response_model=DocumentUploadResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    ingestion_service: DocumentIngestionService = Depends(get_document_ingestion_service),
):
    content = await file.read()
    filename = file.filename or "document.txt"
    document_id = str(uuid4())

    _set_document_status(
        DocumentStatusResponse(
            document_id=document_id,
            filename=filename,
            status="queued",
        )
    )
    background_tasks.add_task(
        _run_document_ingestion_job,
        document_id,
        filename,
        content,
        ingestion_service,
    )

    return DocumentUploadResponse(
        document_id=document_id,
        filename=filename,
        status="queued",
    )


@app.get("/api/v1/documents/{document_id}/status", response_model=DocumentStatusResponse)
async def document_status(document_id: str):
    status = document_ingestion_jobs.get(document_id)
    if status is None:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "document_job_not_found",
                "message": f"Document ingestion job '{document_id}' was not found.",
            },
        )
    return status


@app.get("/health")
async def health():
    return {"status": "ok", "agent_initialized": red_team_agent.initialized}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
