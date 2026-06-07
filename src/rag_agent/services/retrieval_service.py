from typing import Optional
import logging

from rag_agent.core.config import Settings
from rag_agent.storage.vector_store import VectorStore, VectorRecord
from rag_agent.services.llm_service import LLMService
from rag_agent.rag.retriever import cosine_similarity

logger = logging.getLogger(__name__)


class RetrievalService:
    def __init__(
        self,
        settings: Settings,
        vector_store: VectorStore,
        llm_service: LLMService,
    ) -> None:
        self._settings = settings
        self._vector_store = vector_store
        self._llm_service = llm_service

    async def search(self, payload) -> dict:
        result = await self.retrieve_context(payload.query, payload.top_k)
        return {"contexts": result.get("contexts", [])}

    async def retrieve_context(self, query: str, top_k: Optional[int] = None) -> dict:
        try:
            query_embedding = await self._llm_service.embed(query)
            records = self._vector_store.load()

            if not records:
                return {"contexts": []}

            scored = []
            for record in records:
                try:
                    score = cosine_similarity(query_embedding, record.embedding)
                    scored.append((score, record))
                except Exception:
                    continue

            scored.sort(key=lambda x: x[0], reverse=True)

            limit = top_k or self._settings.default_top_k
            contexts = []

            for score, record in scored[:limit]:
                if score < self._settings.min_retrieval_score:
                    continue

                contexts.append({
                    "text": record.text,
                    "source": record.metadata.get("filename", record.filename),
                    "chunk_id": record.chunk_id,
                    "score": round(score, 4),
                })

            return {"contexts": contexts}

        except Exception as e:
            logger.error(f"Retrieval error: {e}")
            return {"contexts": []}


# Создание экземпляра сервиса
_settings = get_settings()  # нужно импортировать
_vector_store = VectorStore(_settings.index_path)
_llm_service = LLMService(...)  # зависит от твоей реализации

retrieval_service = RetrievalService(
    settings=_settings,
    vector_store=_vector_store,
    llm_service=_llm_service,
)
