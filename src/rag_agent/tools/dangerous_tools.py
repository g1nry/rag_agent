import os
import subprocess
from typing import Any
import logging

from .base import BaseTool, ToolMetadata

logger = logging.getLogger(__name__)


class ShellExecuteTool(BaseTool):
    def __init__(self):
        self.metadata = ToolMetadata(
            name="shell_execute",
            description="Выполняет shell-команду. ОЧЕНЬ ОПАСНО.",
            risk_level="high",
            requires_confirmation=True,
        )

    async def arun(self, command: str = "", timeout: int = 30, **kwargs) -> str:
        command = command or kwargs.get("command", "")


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