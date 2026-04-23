from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    use_rag: bool = True
    top_k: int = Field(default=4, ge=1, le=20)


class ChatResponse(BaseModel):
    answer: str
    contexts: list[str] = []


class RetrievalRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=4, ge=1, le=20)


class RetrievalResponse(BaseModel):
    contexts: list[str]


class DocumentIngestResponse(BaseModel):
    filename: str
    chunks_indexed: int


class RetrievalResult(BaseModel):
    contexts: list[str]

