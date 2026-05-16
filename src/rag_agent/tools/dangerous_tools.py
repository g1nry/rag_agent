import os
import subprocess
import logging
from .base import BaseTool, ToolMetadata

logger = logging.getLogger(__name__)


def _safe_str(text: str) -> str:
    """Безопасное преобразование в UTF-8"""
    if isinstance(text, bytes):
        try:
            return text.decode("utf-8", errors="replace")
        except:
            return text.decode("latin-1", errors="replace")
    return str(text)


class ShellExecuteTool(BaseTool):
    def __init__(self):
        self.metadata = ToolMetadata(
            name="shell_execute",
            description="Выполняет shell-команду. ОЧЕНЬ ОПАСНО.",
            risk_level="high",
            requires_confirmation=True,
            category="system"
        )

    async def arun(self, command: str = "", timeout: int = 30, **kwargs) -> str:
        command = command or kwargs.get("command", "")
        try:
            result = subprocess.run(
                command, shell=True, capture_output=True, text=False, timeout=timeout
            )
            stdout = _safe_str(result.stdout)
            stderr = _safe_str(result.stderr)
            output = stdout + stderr

            if result.returncode == 0:
                return f"✅ Выполнено:\n{output}"
            else:
                return f"❌ Ошибка (код {result.returncode}):\n{output}"
        except Exception as e:
            return f"❌ Ошибка выполнения: {str(e)}"


class FileWriteTool(BaseTool):
    def __init__(self):
        self.metadata = ToolMetadata(
            name="file_write",
            description="Записывает текст в файл.",
            risk_level="high",
            requires_confirmation=True,
            category="filesystem"
        )

    async def arun(self, filepath: str = "", content: str = "", **kwargs) -> str:
        filepath = filepath or kwargs.get("filepath", "")
        content = content or kwargs.get("content", "")
        try:
            os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)
            return f"✅ Файл записан: {filepath}"
        except Exception as e:
            return f"❌ Ошибка записи: {str(e)}"


class FileReadTool(BaseTool):
    def __init__(self):
        self.metadata = ToolMetadata(
            name="file_read",
            description="Читает содержимое файла.",
            risk_level="high",
            requires_confirmation=True,
            category="filesystem"
        )

    async def arun(self, filepath: str = "", **kwargs) -> str:
        filepath = filepath or kwargs.get("filepath", "")
        try:
            with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                return f"✅ Содержимое {filepath}:\n{f.read()}"
        except Exception as e:
            return f"❌ Ошибка чтения: {str(e)}"


class FileDeleteTool(BaseTool):
    def __init__(self):
        self.metadata = ToolMetadata(
            name="file_delete",
            description="Удаляет файл или папку. ОЧЕНЬ ОПАСНО.",
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
            return f"✅ Удалено: {filepath}"
        except Exception as e:
            return f"❌ Ошибка удаления: {str(e)}"


class MkdirTool(BaseTool):
    def __init__(self):
        self.metadata = ToolMetadata(
            name="mkdir",
            description="Создаёт директорию.",
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
            return f"❌ Ошибка: {str(e)}"


class ChmodTool(BaseTool):
    def __init__(self):
        self.metadata = ToolMetadata(
            name="chmod",
            description="Меняет права доступа (например 777).",
            risk_level="high",
            requires_confirmation=True,
            category="filesystem"
        )

    async def arun(self, filepath: str = "", mode: str = "", **kwargs) -> str:
        filepath = filepath or kwargs.get("filepath", "")
        mode = mode or kwargs.get("mode", "")
        try:
            os.chmod(filepath, int(mode, 8))
            return f"✅ Права {filepath} → {mode}"
        except Exception as e:
            return f"❌ Ошибка: {str(e)}"


def create_dangerous_tools():
    return [
        ShellExecuteTool(),
        FileWriteTool(),
        FileReadTool(),
        FileDeleteTool(),
        MkdirTool(),
        ChmodTool(),
    ]