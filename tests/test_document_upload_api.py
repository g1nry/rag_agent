from fastapi.testclient import TestClient

from rag_agent.domain.schemas import DocumentIngestResponse
from rag_agent.main import app, get_document_ingestion_service


class FakeDocumentIngestionService:
    def __init__(self) -> None:
        self.filename = ""
        self.content = b""

    async def ingest_document(self, filename: str, content: bytes) -> DocumentIngestResponse:
        self.filename = filename
        self.content = content
        return DocumentIngestResponse(filename=filename, chunks_indexed=1)


def test_chat_v1_document_upload_endpoint_accepts_file() -> None:
    fake_service = FakeDocumentIngestionService()
    app.dependency_overrides[get_document_ingestion_service] = lambda: fake_service

    try:
        client = TestClient(app)
        response = client.post(
            "/chat/v1/documents/upload",
            files={"file": ("notes.txt", b"hello rag", "text/plain")},
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json() == {"filename": "notes.txt", "chunks_indexed": 1}
    assert fake_service.filename == "notes.txt"
    assert fake_service.content == b"hello rag"
