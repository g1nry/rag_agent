from rag_agent.core.config import Settings
from rag_agent.domain.schemas import (
    ContextItem,
    DocumentIngestResponse,
    RetrievalRequest,
    RetrievalResponse,
    RetrievalResult,
)
from rag_agent.rag.chunking import split_text
from rag_agent.rag.retriever import cosine_similarity
from rag_agent.services.llm_service import LLMService
from rag_agent.storage.document_store import DocumentStore
from rag_agent.storage.vector_store import JsonVectorStore, VectorRecord


class RAGService:
    def __init__(
        self,
        settings: Settings,
        document_store: DocumentStore,
        vector_store: JsonVectorStore,
        llm_service: LLMService,
    ) -> None:
        self._settings = settings
        self._document_store = document_store
        self._vector_store = vector_store
        self._llm_service = llm_service

    async def ingest_document(self, filename: str, content: bytes) -> DocumentIngestResponse:
        text = content.decode("utf-8")
        stored_filename = self._document_store.save(filename, text)

        chunks = split_text(
            text=text,
            chunk_size=self._settings.max_chunk_size,
            overlap=self._settings.chunk_overlap,
        )

        records: list[VectorRecord] = []
        for index, chunk in enumerate(chunks):
            embedding = await self._llm_service.embed(chunk)
            records.append(
                VectorRecord(
                    chunk_id=f"{stored_filename}:{index}",
                    text=chunk,
                    embedding=embedding,
                    metadata={
                        "filename": stored_filename,
                        "chunk_index": index,
                    },
                )
            )

        self._vector_store.replace_document(stored_filename, records)
        return DocumentIngestResponse(filename=stored_filename, chunks_indexed=len(records))

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

