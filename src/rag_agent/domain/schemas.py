from pydantic import BaseModel
from typing import List, Optional


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
