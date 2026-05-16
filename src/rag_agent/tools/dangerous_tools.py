import os
import subprocess
from typing import Any
import logging

from .base import BaseTool, ToolMetadata

logger = logging.getLogger(__name__)


class ShellExecuteTool(BaseTool):
    """Опасный инструмент: выполнение shell-команд"""
    
    def __init__(self):
        self.metadata = ToolMetadata(
            name="shell_execute",
            description="Выполняет произвольную shell-команду в системе. ОЧЕНЬ ОПАСНО.",
            risk_level="high",
            requires_confirmation=True,
            category="system"
        )

    async def arun(self, command: str = "", timeout: int = 30, **kwargs) -> str:
        command = command or kwargs.get("command", "")
        
        # === ЛОГИ ===
        import logging, os
        logger = logging.getLogger(__name__)
        logger.warning(f"🛠️  ShellExecuteTool called")
        logger.warning(f"   Command: {command}")
        logger.warning(f"   CWD: {os.getcwd()}")
        # === КОНЕЦ ЛОГОВ ===
        
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=os.getcwd()
            )
            
            output = (result.stdout or "") + (result.stderr or "")
            
            # === ЛОГИ ===
            logger.warning(f"   Return code: {result.returncode}")
            logger.warning(f"   Output length: {len(output)} chars")
            # === КОНЕЦ ЛОГОВ ===
            
            if result.returncode == 0:
                return f"✅ Команда выполнена успешно:\n{output}"
            else:
                return f"❌ Команда завершилась с ошибкой (code {result.returncode}):\n{output}"
                
        except subprocess.TimeoutExpired:
            return "❌ Команда превысила время выполнения (timeout)"
        except Exception as e:
            logger.error(f"Shell execution error: {e}")
            return f"Ошибка выполнения команды: {str(e)}"


class FileWriteTool(BaseTool):
    def __init__(self):
        self.metadata = ToolMetadata(
            name="file_write",
            description="Записывает текст в файл. Высокий риск.",
            risk_level="high",
            requires_confirmation=True,
        )

    async def arun(self, filepath: str = "", content: str = "", **kwargs) -> str:
        filepath = filepath or kwargs.get("filepath", "")
        content = content or kwargs.get("content", "")

# Фабрика для регистрации
def create_dangerous_tools():
    return [
        ShellExecuteTool(),
        FileWriteTool(),
    ]