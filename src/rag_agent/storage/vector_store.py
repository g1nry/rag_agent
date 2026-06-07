import json
import sqlite3
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from rag_agent.storage.errors import IndexLoadError, IndexWriteError


class VectorRecord(BaseModel):
    chunk_id: str
    text: str
    embedding: list[float]
    metadata: dict[str, Any] = Field(default_factory=dict)


class SQLiteVectorStore:
    def __init__(self, index_path: Path) -> None:
        self._legacy_json_path = index_path if index_path.suffix == ".json" else None
        self._index_path = self._sqlite_path(index_path)
        self._index_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()
        self._migrate_legacy_json()

    def load(self) -> list[VectorRecord]:
        try:
            with self._connect() as connection:
                rows = connection.execute(
                    """
                    SELECT chunk_id, text, embedding, metadata
                    FROM vector_records
                    ORDER BY filename, chunk_index, chunk_id
                    """
                ).fetchall()

            return [
                VectorRecord(
                    chunk_id=row["chunk_id"],
                    text=row["text"],
                    embedding=json.loads(row["embedding"]),
                    metadata=json.loads(row["metadata"]),
                )
                for row in rows
            ]
        except IndexLoadError:
            raise
        except ValidationError as exc:
            raise IndexLoadError("Vector index contains records in an invalid format.") from exc
        except (OSError, sqlite3.Error, json.JSONDecodeError) as exc:
            raise IndexLoadError("Vector index could not be read from disk.") from exc

    def add_many(self, records: list[VectorRecord]) -> None:
        self._write_records(records)

    def replace_document(self, filename: str, records: list[VectorRecord]) -> None:
        try:
            with self._connect() as connection:
                connection.execute(
                    "DELETE FROM vector_records WHERE filename = ?",
                    (filename,),
                )
                self._insert_records(connection, records)
                connection.commit()
        except (OSError, sqlite3.Error) as exc:
            raise IndexWriteError("Vector index could not be written.") from exc

    def _write_records(self, records: list[VectorRecord]) -> None:
        try:
            with self._connect() as connection:
                self._insert_records(connection, records)
                connection.commit()
        except (OSError, sqlite3.Error) as exc:
            raise IndexWriteError("Vector index could not be written.") from exc

    def _insert_records(
        self,
        connection: sqlite3.Connection,
        records: list[VectorRecord],
    ) -> None:
        connection.executemany(
            """
            INSERT INTO vector_records (
                chunk_id,
                filename,
                chunk_index,
                text,
                embedding,
                metadata
            )
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(chunk_id) DO UPDATE SET
                filename = excluded.filename,
                chunk_index = excluded.chunk_index,
                text = excluded.text,
                embedding = excluded.embedding,
                metadata = excluded.metadata
            """,
            [
                (
                    record.chunk_id,
                    record.metadata.get("filename", "unknown"),
                    int(record.metadata.get("chunk_index", self._chunk_index(record))),
                    record.text,
                    json.dumps(record.embedding, ensure_ascii=False),
                    json.dumps(record.metadata, ensure_ascii=False),
                )
                for record in records
            ],
        )

    def _ensure_schema(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS vector_records (
                    chunk_id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    embedding TEXT NOT NULL,
                    metadata TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_vector_records_document
                ON vector_records(filename, chunk_index)
                """
            )
            connection.commit()

    def _migrate_legacy_json(self) -> None:
        if self._legacy_json_path is None or not self._legacy_json_path.exists():
            return

        try:
            with self._connect() as connection:
                count = connection.execute(
                    "SELECT COUNT(*) FROM vector_records"
                ).fetchone()[0]
            if count:
                return

            raw_data = json.loads(self._legacy_json_path.read_text(encoding="utf-8"))
            if not isinstance(raw_data, list):
                return

            records = [VectorRecord.model_validate(item) for item in raw_data]
            self._write_records(records)
        except (OSError, sqlite3.Error, json.JSONDecodeError, ValidationError) as exc:
            raise IndexLoadError("Legacy JSON vector index could not be migrated.") from exc

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._index_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _sqlite_path(self, index_path: Path) -> Path:
        if index_path.suffix == ".sqlite3":
            return index_path
        return index_path.with_suffix(".sqlite3")

    def _chunk_index(self, record: VectorRecord) -> int:
        try:
            return int(record.chunk_id.rsplit(":", 1)[-1])
        except ValueError:
            return 0


JsonVectorStore = SQLiteVectorStore
