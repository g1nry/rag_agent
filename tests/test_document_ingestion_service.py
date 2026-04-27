import asyncio
from pathlib import Path

import pytest

from rag_agent.core.config import Settings
from rag_agent.services.document_errors import (
    DocumentEncodingError,
    DocumentTooLargeError,
    EmptyDocumentError,
    UnsupportedDocumentTypeError,
)
from rag_agent.services.document_ingestion_service import DocumentIngestionService
from rag_agent.storage.document_store import DocumentStore
from rag_agent.storage.vector_store import JsonVectorStore


class FakeLLMService:
    async def embed(self, text: str) -> list[float]:
        return [float(len(text)), 1.0]


def build_settings(tmp_path: Path) -> Settings:
    return Settings(
        APP_NAME="rag-agent",
        APP_HOST="127.0.0.1",
        APP_PORT=8000,
        OLLAMA_BASE_URL="http://localhost:11434",
        OLLAMA_CHAT_MODEL="llama3.2:1b",
        OLLAMA_EMBEDDING_MODEL="nomic-embed-text",
        DATA_DIR=tmp_path,
        DOCUMENTS_DIR=tmp_path / "documents",
        INDEX_PATH=tmp_path / "indexes" / "vector_index.json",
        MAX_CHUNK_SIZE=50,
        CHUNK_OVERLAP=10,
        DEFAULT_TOP_K=4,
        MIN_RETRIEVAL_SCORE=0.0,
        MAX_UPLOAD_SIZE_BYTES=10,
        ALLOWED_DOCUMENT_EXTENSIONS=[".txt", ".md"],
    )


def build_service(tmp_path: Path) -> DocumentIngestionService:
    settings = build_settings(tmp_path)
    return DocumentIngestionService(
        settings=settings,
        document_store=DocumentStore(settings.documents_dir),
        vector_store=JsonVectorStore(settings.index_path),
        llm_service=FakeLLMService(),
    )


def test_ingest_rejects_empty_document(tmp_path: Path) -> None:
    service = build_service(tmp_path)

    with pytest.raises(EmptyDocumentError):
        asyncio.run(service.ingest_document("notes.txt", b""))


def test_ingest_rejects_large_document(tmp_path: Path) -> None:
    service = build_service(tmp_path)

    with pytest.raises(DocumentTooLargeError):
        asyncio.run(service.ingest_document("notes.txt", b"01234567890"))


def test_ingest_rejects_unsupported_extension(tmp_path: Path) -> None:
    service = build_service(tmp_path)

    with pytest.raises(UnsupportedDocumentTypeError):
        asyncio.run(service.ingest_document("notes.pdf", b"hello"))


def test_ingest_rejects_non_utf8_content(tmp_path: Path) -> None:
    service = build_service(tmp_path)

    with pytest.raises(DocumentEncodingError):
        asyncio.run(service.ingest_document("notes.txt", b"\xff\xfe\xfd"))
