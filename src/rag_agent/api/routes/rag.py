from fastapi import APIRouter, Depends

from rag_agent.api.routes.dependencies import get_rag_service
from rag_agent.domain.schemas import RetrievalRequest, RetrievalResponse
from rag_agent.services.rag_service import RAGService

router = APIRouter()


@router.post("/search", response_model=RetrievalResponse)
async def search(
    payload: RetrievalRequest,
    rag_service: RAGService = Depends(get_rag_service),
) -> RetrievalResponse:
    return await rag_service.search(payload)

