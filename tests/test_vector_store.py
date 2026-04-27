import json
from pathlib import Path

import pytest

from rag_agent.storage.errors import IndexLoadError
from rag_agent.storage.vector_store import JsonVectorStore, VectorRecord


def test_load_raises_structured_error_for_invalid_json(tmp_path: Path) -> None:
    index_path = tmp_path / "indexes" / "vector_index.json"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text("{not-valid-json", encoding="utf-8")

    store = JsonVectorStore(index_path)

    with pytest.raises(IndexLoadError, match="invalid JSON"):
        store.load()


def test_write_replaces_index_atomically_without_temp_file_leftovers(tmp_path: Path) -> None:
    index_path = tmp_path / "indexes" / "vector_index.json"
    store = JsonVectorStore(index_path)
    records = [
        VectorRecord(
            chunk_id="doc.txt:0",
            text="hello",
            embedding=[1.0, 2.0],
            metadata={"filename": "doc.txt"},
        )
    ]

    store.add_many(records)

    payload = json.loads(index_path.read_text(encoding="utf-8"))
    assert payload[0]["chunk_id"] == "doc.txt:0"
    assert not (index_path.parent / "vector_index.json.tmp").exists()
