from rag_agent.core.config import Settings
from rag_agent.domain.schemas import ContextItem, RetrievalRequest, RetrievalResponse, RetrievalResult
from rag_agent.rag.retriever import cosine_similarity
from rag_agent.services.llm_service import LLMService
from rag_agent.storage.vector_store import JsonVectorStore
from rag_agent.core.config import get_settings
from rag_agent.storage.vector_store import JsonVectorStore
from rag_agent.services.llm_service import LLMService


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

        scored = [
            (cosine_similarity(query_embedding, record.embedding), record)
            for record in records
        ]
        scored.sort(key=lambda item: item[0], reverse=True)

        limit = top_k or self._settings.default_top_k
        contexts: list[ContextItem] = []

        for score, record in scored[:limit]:
            if score < self._settings.min_retrieval_score:
                continue

            contexts.append(
                ContextItem(
                    text=record.text,
                    source=record.metadata.get("filename", "unknown"),
                    chunk_id=record.chunk_id,
                )
            )

        return RetrievalResult(contexts=contexts)

_settings = get_settings()
_vector_store = JsonVectorStore(_settings.index_path)
_llm_service = LLMService(_settings)

retrieval_service = RetrievalService(
    settings=_settings,
    vector_store=_vector_store,
    llm_service=_llm_service,
)
