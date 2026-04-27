import asyncio
from pathlib import Path

from rag_agent.core.config import Settings
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
    )


def test_reingest_replaces_existing_document_chunks(tmp_path: Path) -> None:
    settings = build_settings(tmp_path)
    document_store = DocumentStore(settings.documents_dir)
    vector_store = JsonVectorStore(settings.index_path)
    llm_service = FakeLLMService()

    ingestion_service = DocumentIngestionService(
        settings=settings,
        document_store=document_store,
        vector_store=vector_store,
        llm_service=llm_service,
    )

    first_content = ("alpha " * 30).encode("utf-8")
    second_content = "beta".encode("utf-8")

    asyncio.run(ingestion_service.ingest_document("notes.txt", first_content))
    first_records = vector_store.load()

    asyncio.run(ingestion_service.ingest_document("notes.txt", second_content))
    second_records = vector_store.load()

    assert len(first_records) > 1
    assert len(second_records) == 1
    assert second_records[0].chunk_id == "notes.txt:0"
    assert second_records[0].text == "beta"
    assert all(record.metadata.get("filename") == "notes.txt" for record in second_records)
