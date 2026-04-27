import asyncio
from pathlib import Path

from rag_agent.core.config import Settings
from rag_agent.services.retrieval_service import RetrievalService
from rag_agent.storage.vector_store import JsonVectorStore, VectorRecord


class FakeLLMService:
    async def embed(self, text: str) -> list[float]:
        if text == "alpha":
            return [1.0, 0.0]
        if text == "beta":
            return [0.0, 1.0]
        return [1.0, 0.0]


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
        MIN_RETRIEVAL_SCORE=0.2,
    )


def test_retrieval_returns_ranked_contexts(tmp_path: Path) -> None:
    settings = build_settings(tmp_path)
    vector_store = JsonVectorStore(settings.index_path)
    vector_store.add_many(
        [
            VectorRecord(
                chunk_id="doc.txt:0",
                text="alpha text",
                embedding=[1.0, 0.0],
                metadata={"filename": "doc.txt"},
            ),
            VectorRecord(
                chunk_id="doc.txt:1",
                text="beta text",
                embedding=[0.0, 1.0],
                metadata={"filename": "doc.txt"},
            ),
        ]
    )
    retrieval_service = RetrievalService(
        settings=settings,
        vector_store=vector_store,
        llm_service=FakeLLMService(),
    )

    result = asyncio.run(retrieval_service.retrieve_context("alpha", top_k=2))

    assert len(result.contexts) == 1
    assert result.contexts[0].chunk_id == "doc.txt:0"
    assert result.contexts[0].source == "doc.txt"
    assert result.contexts[0].text == "alpha text"
