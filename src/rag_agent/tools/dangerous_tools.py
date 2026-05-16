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
            description="Выполняет произвольную shell-команду в системе. ОЧЕНЬ ОПАСНО. Используй только если точно понимаешь последствия.",
            risk_level="high",
            requires_confirmation=True,
            category="system"
        )

    async def arun(self, command: str, timeout: int = 30, **kwargs) -> str:
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=os.getcwd()
            )
            
            output = result.stdout + result.stderr
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
    """Опасный инструмент: запись в файл"""
    
    def __init__(self):
        self.metadata = ToolMetadata(
            name="file_write",
            description="Записывает текст в файл (перезаписывает если существует). Высокий риск.",
            risk_level="high",
            requires_confirmation=True,
            category="filesystem"
        )

    async def arun(self, filepath: str, content: str, **kwargs) -> str:
        try:
            # Безопасность: запрещаем запись вне текущей директории проекта (можно расширить)
            if ".." in filepath or filepath.startswith("/"):
                return "❌ Запрещено записывать файлы вне рабочей директории"
            
            os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
            
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)
            
            return f"✅ Файл успешно записан: {filepath} ({len(content)} символов)"
        except Exception as e:
            return f"❌ Ошибка записи файла: {str(e)}"


# Фабрика для регистрации
def create_dangerous_tools():
    return [
        ShellExecuteTool(),
        FileWriteTool(),
    ]