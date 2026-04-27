from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from rag_agent.api.router import api_router
from rag_agent.core.config import get_settings
from rag_agent.integrations.ollama.exceptions import OllamaError
from rag_agent.services.document_errors import DocumentIngestionError


@asynccontextmanager
async def lifespan(_: FastAPI):
    settings = get_settings()
    settings.documents_dir.mkdir(parents=True, exist_ok=True)
    settings.index_path.parent.mkdir(parents=True, exist_ok=True)
    yield


settings = get_settings()
frontend_dir = Path(__file__).resolve().parents[2] / "frontend"

app = FastAPI(title=settings.app_name, lifespan=lifespan)
app.include_router(api_router, prefix="/api/v1")
if settings.ui_enabled:
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")


@app.exception_handler(OllamaError)
async def handle_ollama_error(_: Request, exc: OllamaError) -> JSONResponse:
    payload = {
        "detail": exc.detail,
        "error_code": exc.error_code,
    }
    if exc.upstream_status_code is not None:
        payload["upstream_status_code"] = exc.upstream_status_code
    return JSONResponse(status_code=exc.status_code, content=payload)


@app.exception_handler(DocumentIngestionError)
async def handle_document_ingestion_error(_: Request, exc: DocumentIngestionError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "detail": exc.detail,
            "error_code": exc.error_code,
        },
    )


if settings.ui_enabled:
    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(frontend_dir / "index.html")
else:
    @app.get("/", include_in_schema=False)
    async def index() -> dict[str, str]:
        return {"message": "UI is disabled in config.toml"}


@app.get("/health", tags=["system"])
async def healthcheck() -> dict[str, str]:
    return {"status": "ok"}
