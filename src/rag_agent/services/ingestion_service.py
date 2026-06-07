import asyncio
from typing import List, Dict, Optional
import logging
from pathlib import Path

from ..core.config import get_settings
from .retrieval_service import retrieval_service

logger = logging.getLogger(__name__)


class IngestionService:
    """Сервис для асинхронной загрузки и индексации документов"""
    
    def __init__(self):
        self.settings = get_settings()
        self._lock = asyncio.Lock()

    async def ingest_document(self, file_path: str, filename: str) -> Dict:
        """Асинхронная индексация одного документа"""
        async with self._lock:
            try:
                logger.info(f"📄 Начало индексации: {filename}")
                
                # Здесь будет улучшенный chunking (Markdown-aware)
                # Пока используем существующий retrieval_service
                result = await retrieval_service.ingest_file(file_path, filename)
                
                logger.info(f"✅ Документ успешно проиндексирован: {filename}")
                return {
                    "document_id": filename,
                    "status": "success",
                    "chunks_count": result.get("chunks_count", 0) if isinstance(result, dict) else 0
                }
                
            except Exception as e:
                logger.error(f"❌ Ошибка индексации {filename}: {e}")
                return {
                    "document_id": filename,
                    "status": "error",
                    "error": str(e)
                }


ingestion_service = IngestionService()