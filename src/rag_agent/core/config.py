import os
import tomllib
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import AnyHttpUrl, BaseModel, ConfigDict, Field, model_validator


DEFAULT_CONFIG_PATH = Path("config.toml")
ENV_TO_FIELD = {
    "APP_NAME": "app_name",
    "APP_HOST": "app_host",
    "APP_PORT": "app_port",
    "OLLAMA_BASE_URL": "ollama_base_url",
    "OLLAMA_CHAT_MODEL": "ollama_chat_model",
    "OLLAMA_EMBEDDING_MODEL": "ollama_embedding_model",
    "OLLAMA_TIMEOUT": "ollama_timeout",
    "DATA_DIR": "data_dir",
    "DOCUMENTS_DIR": "documents_dir",
    "INDEX_PATH": "index_path",
    "MAX_CHUNK_SIZE": "max_chunk_size",
    "CHUNK_OVERLAP": "chunk_overlap",
    "DEFAULT_TOP_K": "default_top_k",
    "MIN_RETRIEVAL_SCORE": "min_retrieval_score",
    "UI_ENABLED": "ui_enabled",
    "MAX_UPLOAD_SIZE_BYTES": "max_upload_size_bytes",
    "ALLOWED_DOCUMENT_EXTENSIONS": "allowed_document_extensions",
}


class Settings(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    app_name: str = Field(default="rag-agent", alias="APP_NAME")
    app_host: str = Field(default="0.0.0.0", alias="APP_HOST")
    app_port: int = Field(default=8000, alias="APP_PORT")
    ollama_base_url: AnyHttpUrl = Field(default="http://localhost:11434", alias="OLLAMA_BASE_URL")
    ollama_chat_model: str = Field(default="llama3.2:1b", alias="OLLAMA_CHAT_MODEL")
    ollama_embedding_model: str = Field(default="nomic-embed-text", alias="OLLAMA_EMBEDDING_MODEL")
    ollama_timeout: float = Field(default=60.0, alias="OLLAMA_TIMEOUT")
    data_dir: Path = Field(default=Path("./data"), alias="DATA_DIR")
    documents_dir: Path | None = Field(default=None, alias="DOCUMENTS_DIR")
    index_path: Path | None = Field(default=None, alias="INDEX_PATH")
    max_chunk_size: int = Field(default=800, alias="MAX_CHUNK_SIZE")
    chunk_overlap: int = Field(default=120, alias="CHUNK_OVERLAP")
    default_top_k: int = Field(default=4, alias="DEFAULT_TOP_K")
    min_retrieval_score: float = Field(default=0.2, alias="MIN_RETRIEVAL_SCORE")
    ui_enabled: bool = Field(default=True, alias="UI_ENABLED")
    max_upload_size_bytes: int = Field(default=1_048_576, alias="MAX_UPLOAD_SIZE_BYTES")
    allowed_document_extensions: list[str] = Field(
        default_factory=lambda: [".txt", ".md"],
        alias="ALLOWED_DOCUMENT_EXTENSIONS",
    )

    @model_validator(mode="after")
    def fill_derived_paths(self) -> "Settings":
        if self.documents_dir is None:
            self.documents_dir = self.data_dir / "documents"
        if self.index_path is None:
            self.index_path = self.data_dir / "indexes" / "vector_index.json"
        self.allowed_document_extensions = [
            extension.lower() if extension.startswith(".") else f".{extension.lower()}"
            for extension in self.allowed_document_extensions
        ]
        return self


def _resolve_path(value: str, base_dir: Path) -> str:
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str((base_dir / path).resolve())


def _load_config_file(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        return {}

    with config_path.open("rb") as file:
        raw_config = tomllib.load(file)

    base_dir = config_path.parent.resolve()
    app = raw_config.get("app", {})
    ollama = raw_config.get("ollama", {})
    data = raw_config.get("data", {})
    rag = raw_config.get("rag", {})
    ui = raw_config.get("ui", {})
    documents = raw_config.get("documents", {})

    config: dict[str, Any] = {}

    if "name" in app:
        config["app_name"] = app["name"]
    if "host" in app:
        config["app_host"] = app["host"]
    if "port" in app:
        config["app_port"] = app["port"]

    if "base_url" in ollama:
        config["ollama_base_url"] = ollama["base_url"]
    if "chat_model" in ollama:
        config["ollama_chat_model"] = ollama["chat_model"]
    if "embedding_model" in ollama:
        config["ollama_embedding_model"] = ollama["embedding_model"]
    if "timeout" in ollama:
        config["ollama_timeout"] = ollama["timeout"]

    if "dir" in data:
        config["data_dir"] = _resolve_path(data["dir"], base_dir)
    if "documents_dir" in data:
        config["documents_dir"] = _resolve_path(data["documents_dir"], base_dir)
    if "index_path" in data:
        config["index_path"] = _resolve_path(data["index_path"], base_dir)

    if "max_chunk_size" in rag:
        config["max_chunk_size"] = rag["max_chunk_size"]
    if "chunk_overlap" in rag:
        config["chunk_overlap"] = rag["chunk_overlap"]
    if "default_top_k" in rag:
        config["default_top_k"] = rag["default_top_k"]
    if "min_retrieval_score" in rag:
        config["min_retrieval_score"] = rag["min_retrieval_score"]

    if "enabled" in ui:
        config["ui_enabled"] = ui["enabled"]

    if "max_upload_size_bytes" in documents:
        config["max_upload_size_bytes"] = documents["max_upload_size_bytes"]
    if "allowed_extensions" in documents:
        config["allowed_document_extensions"] = documents["allowed_extensions"]

    return config


def _load_env_overrides() -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for env_name, field_name in ENV_TO_FIELD.items():
        value = os.getenv(env_name)
        if value is not None:
            if env_name == "ALLOWED_DOCUMENT_EXTENSIONS":
                overrides[field_name] = [
                    item.strip()
                    for item in value.split(",")
                    if item.strip()
                ]
                continue
            overrides[field_name] = value
    return overrides


def load_settings(config_path: Path | None = None) -> Settings:
    resolved_path = config_path
    if resolved_path is None:
        resolved_path = Path(os.getenv("RAG_AGENT_CONFIG", DEFAULT_CONFIG_PATH))

    config_values = _load_config_file(resolved_path)
    env_values = _load_env_overrides()
    merged = {**config_values, **env_values}
    return Settings.model_validate(merged)


@lru_cache
def get_settings() -> Settings:
    return load_settings()
