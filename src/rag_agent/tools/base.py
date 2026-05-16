from pydantic import BaseModel, Field
from typing import Literal, Any, Dict
from abc import ABC, abstractmethod


RiskLevel = Literal["safe", "medium", "high"]


class ToolMetadata(BaseModel):
    """Метаданные инструмента для системы безопасности"""
    name: str
    description: str
    risk_level: RiskLevel = "safe"
    requires_confirmation: bool = False
    category: str = "general"


class BaseTool(ABC):
    """Базовый класс для всех инструментов"""
    
    metadata: ToolMetadata

    @abstractmethod
    async def arun(self, **kwargs) -> Any:
        """Асинхронное выполнение инструмента"""
        pass

    def get_metadata(self) -> Dict:
        return self.metadata.model_dump()