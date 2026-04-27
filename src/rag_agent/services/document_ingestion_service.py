from rag_agent.core.config import Settings
from rag_agent.domain.schemas import DocumentIngestResponse
from rag_agent.rag.chunking import split_text
from rag_agent.services.llm_service import LLMService
from rag_agent.storage.document_store import DocumentStore
from rag_agent.storage.vector_store import JsonVectorStore, VectorRecord


class DocumentIngestionService:
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
