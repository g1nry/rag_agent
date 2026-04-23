from fastapi import APIRouter, Depends, File, UploadFile

from rag_agent.api.routes.dependencies import get_rag_service
from rag_agent.domain.schemas import DocumentIngestResponse
from rag_agent.services.rag_service import RAGService

router = APIRouter()


@router.post("/upload", response_model=DocumentIngestResponse)
async def upload_document(
    file: UploadFile = File(...),
    rag_service: RAGService = Depends(get_rag_service),
) -> DocumentIngestResponse:
    content = await file.read()
    return await rag_service.ingest_document(file.filename or "document.txt", content)

