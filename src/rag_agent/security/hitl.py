from typing import Dict, Any, Optional, Callable
import logging

logger = logging.getLogger(__name__)


class HumanInTheLoopManager:
    """Менеджер для Human-in-the-Loop подтверждений"""
    
    def __init__(self, confirmation_callback: Optional[Callable] = None):
        """
        confirmation_callback: функция, которая принимает (tool_name, args, risk_level)
        и возвращает bool (подтверждено ли)
        """
        self.confirmation_callback = confirmation_callback
        self.pending_confirmations: Dict[str, Dict] = {}
    
    def needs_confirmation(self, tool_name: str, risk_level: str) -> bool:
        """Нужно ли подтверждение для этого инструмента"""
        return risk_level in ["medium", "high"]
    
    async def request_confirmation(self, tool_name: str, args: Dict, risk_level: str) -> bool:
        """Запрашивает подтверждение у человека"""
        if self.confirmation_callback:
            try:
                result = await self.confirmation_callback(tool_name, args, risk_level)
                return bool(result)
            except Exception as e:
                logger.error(f"HITL confirmation error: {e}")
                return False
        
        # По умолчанию для high риска — не подтверждаем автоматически
        if risk_level == "high":
            logger.warning(f"⚠️  High-risk tool {tool_name} called without confirmation callback!")
            return False
        
        return True  # Для medium можно разрешить
    
    def set_callback(self, callback: Callable):
        self.confirmation_callback = callback


hitl_manager = HumanInTheLoopManager()