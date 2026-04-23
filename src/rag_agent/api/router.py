from fastapi import APIRouter

from rag_agent.api.routes import chat, documents, rag

api_router = APIRouter()
api_router.include_router(chat.router, tags=["chat"])
api_router.include_router(rag.router, prefix="/rag", tags=["rag"])
api_router.include_router(documents.router, prefix="/documents", tags=["documents"])

