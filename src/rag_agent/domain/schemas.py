from pydantic import BaseModel
from typing import List, Literal, Optional


class ContextItem(BaseModel):
    text: str
    source: str
    chunk_id: str


class RetrievalRequest(BaseModel):
    query: str
    top_k: Optional[int] = None


class RetrievalResult(BaseModel):
    contexts: List[ContextItem]


class RetrievalResponse(BaseModel):
    contexts: List[ContextItem]


class DocumentIngestResponse(BaseModel):
    filename: str
    chunks_indexed: int


DocumentIngestionStatus = Literal["queued", "indexing", "indexed", "failed"]


class DocumentUploadResponse(BaseModel):
    document_id: str
    filename: str
    status: DocumentIngestionStatus


class DocumentStatusResponse(BaseModel):
    document_id: str
    filename: str
    status: DocumentIngestionStatus
    chunks_indexed: Optional[int] = None
    error: Optional[str] = None
    message: Optional[str] = None
