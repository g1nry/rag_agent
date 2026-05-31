from contextlib import asynccontextmanager
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
import logging

from .agents import red_team_agent
from .core.config import get_settings
from .domain.schemas import DocumentIngestResponse
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
document_ingestion_service = DocumentIngestionService(
    settings=settings,
    document_store=DocumentStore(settings.documents_dir),
    vector_store=JsonVectorStore(settings.index_path),
    llm_service=LLMService(
        client=OllamaClient(str(settings.ollama_base_url)),
        settings=settings,
    ),
)


class ChatRequest(BaseModel):
    message: str
    thread_id: str = "default"
    use_rag: bool = True
    top_k: int = 4


def get_document_ingestion_service() -> DocumentIngestionService:
    return document_ingestion_service


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
                <li><code>POST /api/agent/chat</code></li>
                <li><code>GET /health</code></li>
            </ul>
        </body>
    </html>
    """


async def _guarded_chat(request: ChatRequest):
    # === INPUT RAIL ===
    blocked_in = await check_message_guardrails(
        [{"role": "user", "content": request.message}]
    )
    if blocked_in:
        logger.warning(f"Input rail blocked: {request.message[:80]}...")
        return {
            "answer": blocked_in,
            "contexts": [],
            "blocked": True,
            "blocked_by": "input_rail",
        }

    contexts = []
    agent_message = request.message

    if request.use_rag:
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

        if contexts:
            formatted_contexts = "\n\n".join(
                (
                    f"[{index}] source={context.source}, chunk_id={context.chunk_id}\n"
                    f"{context.text}"
                )
                for index, context in enumerate(contexts, start=1)
            )
            agent_message = (
                "Ответь на вопрос пользователя, используя найденный RAG-контекст. "
                "Если контекст не содержит ответа, прямо скажи об этом. "
                "Не утверждай, что файла нет, если контекст ниже был найден.\n\n"
                f"RAG-контекст:\n{formatted_contexts}\n\n"
                f"Вопрос пользователя:\n{request.message}"
            )

    # === Основной вызов агента ===
    result = await red_team_agent.ainvoke(
        message=agent_message,
        thread_id=request.thread_id,
    )
    answer = result.get("response", "")

    # === OUTPUT RAIL (защита от RAG poisoning) ===
    blocked_out = await check_output_guardrails(answer)
    if blocked_out:
        logger.warning("Output rail blocked agent response")
        return {
            "answer": blocked_out,
            "contexts": [],
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


@app.post("/api/v1/documents/upload", response_model=DocumentIngestResponse)
async def upload_document(
    file: UploadFile = File(...),
    ingestion_service: DocumentIngestionService = Depends(get_document_ingestion_service),
):
    content = await file.read()

    try:
        return await ingestion_service.ingest_document(file.filename or "document.txt", content)
    except DocumentIngestionError as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail={
                "error": exc.error_code,
                "message": exc.detail,
            },
        ) from exc


@app.get("/health")
async def health():
    return {"status": "ok", "agent_initialized": red_team_agent.initialized}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
