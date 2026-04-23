from functools import lru_cache
from pathlib import Path

from pydantic import AnyHttpUrl, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    app_name: str = Field(default="rag-agent", alias="APP_NAME")
    app_host: str = Field(default="0.0.0.0", alias="APP_HOST")
    app_port: int = Field(default=8000, alias="APP_PORT")
    ollama_base_url: AnyHttpUrl = Field(alias="OLLAMA_BASE_URL")
    ollama_chat_model: str = Field(default="llama3.1:8b", alias="OLLAMA_CHAT_MODEL")
    ollama_embedding_model: str = Field(default="nomic-embed-text", alias="OLLAMA_EMBEDDING_MODEL")
    data_dir: Path = Field(default=Path("./data"), alias="DATA_DIR")
    documents_dir: Path = Field(default=Path("./data/documents"), alias="DOCUMENTS_DIR")
    index_path: Path = Field(default=Path("./data/indexes/vector_index.json"), alias="INDEX_PATH")
    max_chunk_size: int = Field(default=800, alias="MAX_CHUNK_SIZE")
    chunk_overlap: int = Field(default=120, alias="CHUNK_OVERLAP")
    default_top_k: int = Field(default=4, alias="DEFAULT_TOP_K")

    min_retrieval_score: float = Field(default=0.2, alias="MIN_RETRIEVAL_SCORE")



@lru_cache
def get_settings() -> Settings:
    return Settings()

