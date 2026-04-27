from pathlib import Path

from rag_agent.core.config import Settings
from rag_agent.domain.schemas import DocumentIngestResponse
from rag_agent.rag.chunking import split_text
from rag_agent.services.document_errors import (
    DocumentEncodingError,
    DocumentTooLargeError,
    EmptyDocumentError,
    UnsupportedDocumentTypeError,
)
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
        self._validate_file(filename, content)

        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise DocumentEncodingError(filename) from exc

        if not text.strip():
            raise EmptyDocumentError()

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

    def _validate_file(self, filename: str, content: bytes) -> None:
        if not content:
            raise EmptyDocumentError()

        if len(content) > self._settings.max_upload_size_bytes:
            raise DocumentTooLargeError(self._settings.max_upload_size_bytes)

        extension = Path(filename).suffix.lower()
        if extension not in self._settings.allowed_document_extensions:
            raise UnsupportedDocumentTypeError(filename)
