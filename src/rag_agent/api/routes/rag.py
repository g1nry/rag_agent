from fastapi import APIRouter, Depends

from rag_agent.api.routes.dependencies import get_retrieval_service
from rag_agent.domain.schemas import RetrievalRequest, RetrievalResponse
from rag_agent.services.retrieval_service import RetrievalService

router = APIRouter()


@router.post("/search", response_model=RetrievalResponse)
async def search(
    payload: RetrievalRequest,
    retrieval_service: RetrievalService = Depends(get_retrieval_service),
) -> RetrievalResponse:
    return await retrieval_service.search(payload)
