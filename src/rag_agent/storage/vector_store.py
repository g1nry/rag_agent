import json
import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from rag_agent.storage.errors import IndexLoadError, IndexWriteError


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
        try:
            if not self._index_path.exists():
                self._index_path.write_text("[]", encoding="utf-8")

            raw_data = json.loads(self._index_path.read_text(encoding="utf-8"))
            if not isinstance(raw_data, list):
                raise IndexLoadError("Vector index has an unexpected JSON structure.")

            return [VectorRecord.model_validate(item) for item in raw_data]
        except IndexLoadError:
            raise
        except json.JSONDecodeError as exc:
            raise IndexLoadError("Vector index contains invalid JSON.") from exc
        except ValidationError as exc:
            raise IndexLoadError("Vector index contains records in an invalid format.") from exc
        except OSError as exc:
            raise IndexLoadError("Vector index could not be read from disk.") from exc

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
        serialized = json.dumps(payload, ensure_ascii=False, indent=2)
        temp_path = self._index_path.with_suffix(f"{self._index_path.suffix}.tmp")

        try:
            temp_path.write_text(serialized, encoding="utf-8")
            os.replace(temp_path, self._index_path)
        except OSError as exc:
            raise IndexWriteError("Vector index could not be written atomically.") from exc
        finally:
            if temp_path.exists():
                temp_path.unlink(missing_ok=True)
