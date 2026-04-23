from fastapi import APIRouter, Depends

from rag_agent.agent.orchestrator import AgentOrchestrator
from rag_agent.api.routes.dependencies import get_agent_orchestrator
from rag_agent.domain.schemas import ChatRequest, ChatResponse

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def chat(
    payload: ChatRequest,
    orchestrator: AgentOrchestrator = Depends(get_agent_orchestrator),
) -> ChatResponse:
    return await orchestrator.reply(payload)

