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

class FileReadTool(BaseTool):
    def __init__(self):
        self.metadata = ToolMetadata(
            name="file_read",
            description="Читает содержимое файла. Может быть опасно, если читать чувствительные данные.",
            risk_level="high",
            requires_confirmation=True,
            category="filesystem"
        )

    async def arun(self, filepath: str = "", **kwargs) -> str:
        filepath = filepath or kwargs.get("filepath", "")
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
            return f"✅ Содержимое файла {filepath}:\n{content}"
        except Exception as e:
            return f"❌ Ошибка чтения файла: {str(e)}"


class FileDeleteTool(BaseTool):
    def __init__(self):
        self.metadata = ToolMetadata(
            name="file_delete",
            description="Удаляет файл или директорию. ОЧЕНЬ ОПАСНО.",
            risk_level="high",
            requires_confirmation=True,
            category="filesystem"
        )

    async def arun(self, filepath: str = "", **kwargs) -> str:
        filepath = filepath or kwargs.get("filepath", "")
        try:
            if os.path.isdir(filepath):
                import shutil
                shutil.rmtree(filepath)
            else:
                os.remove(filepath)
            return f"✅ Файл/директория удалена: {filepath}"
        except Exception as e:
            return f"❌ Ошибка удаления: {str(e)}"


class MkdirTool(BaseTool):
    def __init__(self):
        self.metadata = ToolMetadata(
            name="mkdir",
            description="Создаёт директорию (включая родительские).",
            risk_level="medium",
            requires_confirmation=False,
            category="filesystem"
        )

    async def arun(self, path: str = "", **kwargs) -> str:
        path = path or kwargs.get("path", "")
        try:
            os.makedirs(path, exist_ok=True)
            return f"✅ Директория создана: {path}"
        except Exception as e:
            return f"❌ Ошибка создания директории: {str(e)}"


class ChmodTool(BaseTool):
    def __init__(self):
        self.metadata = ToolMetadata(
            name="chmod",
            description="Меняет права доступа к файлу (например: 755, 644).",
            risk_level="high",
            requires_confirmation=True,
            category="filesystem"
        )

    async def arun(self, filepath: str = "", mode: str = "", **kwargs) -> str:
        filepath = filepath or kwargs.get("filepath", "")
        mode = mode or kwargs.get("mode", "")
        try:
            os.chmod(filepath, int(mode, 8))
            return f"✅ Права изменены: {filepath} → {mode}"
        except Exception as e:
            return f"❌ Ошибка изменения прав: {str(e)}"

# Фабрика для регистрации
def create_dangerous_tools():
    return [
        ShellExecuteTool(),
        FileWriteTool(),
        FileReadTool(),
        FileDeleteTool(),
        MkdirTool(),
        ChmodTool(),
    ]