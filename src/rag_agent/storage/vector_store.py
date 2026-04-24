import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class VectorRecord(BaseModel):
    chunk_id: str
    text: str
    embedding: list[float]
    metadata: dict[str, Any] = Field(default_factory=dict)


class JsonVectorStore:
    def __init__(self, index_path: Path) -> None:
        self._index_path = index_path
        self._index_path.parent.mkdir(parents=True, exist_ok=True)
        if not self._index_path.exists():
            self._index_path.write_text("[]", encoding="utf-8")

    def load(self) -> list[VectorRecord]:
        raw_data = json.loads(self._index_path.read_text(encoding="utf-8"))
        return [VectorRecord.model_validate(item) for item in raw_data]

    def add_many(self, records: list[VectorRecord]) -> None:
        existing = self.load()
        existing.extend(records)
        self._write(existing)

    def replace_document(self, filename: str, records: list[VectorRecord]) -> None:
        existing = self.load()
        filtered = [
            record
            for record in existing
            if record.metadata.get("filename") != filename
        ]
        filtered.extend(records)
        self._write(filtered)

    def _write(self, records: list[VectorRecord]) -> None:
        payload = [record.model_dump(mode="json") for record in records]
        self._index_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
