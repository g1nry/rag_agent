import sqlite3
import json
from pathlib import Path
from typing import List, Optional
from pydantic import BaseModel

from .errors import IndexLoadError, IndexWriteError


class VectorRecord(BaseModel):
    chunk_id: str
    filename: str
    chunk_index: int
    text: str
    embedding: List[float]
    metadata: dict


class VectorStore:
    """SQLite-based vector store with migration from legacy JSON."""

    def __init__(self, index_path: Path):
        self.index_path = self._sqlite_path(index_path)
        self._ensure_schema()
        self._migrate_legacy_json()

    def _sqlite_path(self, path: Path) -> Path:
        if path.suffix == ".json":
            return path.with_suffix(".sqlite3")
        return path

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.index_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self):
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS vector_records (
                    chunk_id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    embedding TEXT NOT NULL,
                    metadata TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_vector_records_document 
                ON vector_records(filename, chunk_index)
            """)

    def _migrate_legacy_json(self):
        json_path = self.index_path.with_suffix(".json")
        if not json_path.exists():
            return

        with self._connect() as conn:
            count = conn.execute("SELECT COUNT(*) FROM vector_records").fetchone()[0]
            if count > 0:
                return  # уже есть данные

        # Миграция
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            records = [VectorRecord.model_validate(item) for item in data]
            self.add_many(records)
            print(f"[VectorStore] Migrated {len(records)} records from {json_path}")
        except Exception as e:
            raise IndexLoadError(f"Failed to migrate legacy JSON: {e}")

    def load(self) -> List[VectorRecord]:
        try:
            with self._connect() as conn:
                rows = conn.execute("""
                    SELECT * FROM vector_records 
                    ORDER BY filename, chunk_index, chunk_id
                """).fetchall()

            records = []
            for row in rows:
                records.append(VectorRecord(
                    chunk_id=row["chunk_id"],
                    filename=row["filename"],
                    chunk_index=row["chunk_index"],
                    text=row["text"],
                    embedding=json.loads(row["embedding"]),
                    metadata=json.loads(row["metadata"]),
                ))
            return records
        except Exception as e:
            raise IndexLoadError(f"Failed to load index: {e}")

    def add_many(self, records: List[VectorRecord]):
        try:
            with self._connect() as conn:
                self._insert_records(conn, records)
                conn.commit()
        except Exception as e:
            raise IndexWriteError(f"Failed to write records: {e}")

    def replace_document(self, filename: str, records: List[VectorRecord]):
        try:
            with self._connect() as conn:
                conn.execute("DELETE FROM vector_records WHERE filename = ?", (filename,))
                self._insert_records(conn, records)
                conn.commit()
        except Exception as e:
            raise IndexWriteError(f"Failed to replace document {filename}: {e}")

    def _insert_records(self, conn: sqlite3.Connection, records: List[VectorRecord]):
        data = []
        for rec in records:
            filename = rec.metadata.get("filename", rec.filename)
            chunk_index = rec.metadata.get("chunk_index", self._chunk_index(rec))
            data.append((
                rec.chunk_id,
                filename,
                chunk_index,
                rec.text,
                json.dumps(rec.embedding),
                json.dumps(rec.metadata),
            ))

        conn.executemany("""
            INSERT INTO vector_records (chunk_id, filename, chunk_index, text, embedding, metadata)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(chunk_id) DO UPDATE SET
                filename=excluded.filename,
                chunk_index=excluded.chunk_index,
                text=excluded.text,
                embedding=excluded.embedding,
                metadata=excluded.metadata
        """, data)

    def _chunk_index(self, record: VectorRecord) -> int:
        try:
            return int(record.chunk_id.split(":")[-1])
        except:
            return 0


# Для обратной совместимости (можно потом убрать)
JsonVectorStore = VectorStore