from fastapi import APIRouter, Depends, File, UploadFile

from rag_agent.api.routes.dependencies import get_document_ingestion_service
from rag_agent.domain.schemas import DocumentIngestResponse
from rag_agent.services.document_ingestion_service import DocumentIngestionService

router = APIRouter()


@router.post("/upload", response_model=DocumentIngestResponse)
async def upload_document(
    file: UploadFile = File(...),
    ingestion_service: DocumentIngestionService = Depends(get_document_ingestion_service),
) -> DocumentIngestResponse:
    content = await file.read()
    return await ingestion_service.ingest_document(file.filename or "document.txt", content)
