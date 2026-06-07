from pathlib import Path

from rag_agent.core.config import Settings
from rag_agent.domain.schemas import ContextItem, RetrievalRequest, RetrievalResponse, RetrievalResult
from rag_agent.rag.retriever import cosine_similarity
from rag_agent.services.llm_service import LLMService
from rag_agent.storage.vector_store import JsonVectorStore
from rag_agent.storage.vector_store import VectorRecord
from rag_agent.core.config import get_settings
from rag_agent.integrations.ollama.client import OllamaClient



class RetrievalService:
    def __init__(
        self,
        settings: Settings,
        vector_store: JsonVectorStore,
        llm_service: LLMService,
    ) -> None:
        self._settings = settings
        self._vector_store = vector_store
        self._llm_service = llm_service

    async def search(self, payload: RetrievalRequest) -> RetrievalResponse:
        result = await self.retrieve_context(payload.query, payload.top_k)
        return RetrievalResponse(contexts=result.contexts)

    async def retrieve_context(self, query: str, top_k: int | None = None) -> RetrievalResult:
        query_embedding = await self._llm_service.embed(query)
        records = self._vector_store.load()
        limit = top_k or self._settings.default_top_k

        scored = [
            (cosine_similarity(query_embedding, record.embedding), record)
            for record in records
        ]
        scored.sort(key=lambda item: item[0], reverse=True)

        contexts: list[ContextItem] = []

        for score, record in scored[:limit]:
            if score < self._settings.min_retrieval_score:
                continue

            contexts.append(self._record_to_context(record))

        if not contexts:
            contexts = self._fallback_contexts(query, records, limit)

        return RetrievalResult(contexts=contexts)

    def _fallback_contexts(
        self,
        query: str,
        records: list[VectorRecord],
        limit: int,
    ) -> list[ContextItem]:
        if not records:
            return []

        query_lower = query.lower()
        sources = {
            record.metadata.get("filename", "unknown")
            for record in records
        }

        matching_records = [
            record
            for record in records
            if record.metadata.get("filename", "").lower() in query_lower
        ]

        if not matching_records and len(sources) == 1:
            matching_records = records

        matching_records.sort(
            key=lambda record: (
                record.metadata.get("filename", ""),
                record.metadata.get("chunk_index", self._chunk_index(record)),
            )
        )

        return [
            self._record_to_context(record)
            for record in matching_records[:limit]
        ]

    def _record_to_context(self, record: VectorRecord) -> ContextItem:
        return ContextItem(
            text=record.text,
            source=record.metadata.get("filename", "unknown"),
            chunk_id=record.chunk_id,
        )

    def _chunk_index(self, record: VectorRecord) -> int:
        try:
            return int(Path(record.chunk_id).name.rsplit(":", 1)[-1])
        except ValueError:
            return 0

_settings = get_settings()
_vector_store = JsonVectorStore(_settings.index_path)
_ollama_client = OllamaClient(str(_settings.ollama_base_url))
_llm_service = LLMService(client=_ollama_client, settings=_settings)

retrieval_service = RetrievalService(
    settings=_settings,
    vector_store=_vector_store,
    llm_service=_llm_service,
)
