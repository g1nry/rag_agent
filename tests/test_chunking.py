import pytest

from rag_agent.rag.chunking import split_text


def test_split_text_uses_overlap() -> None:
    assert split_text("abcdefghij", chunk_size=4, overlap=1) == [
        "abcd",
        "defg",
        "ghij",
    ]


def test_split_text_rejects_invalid_overlap() -> None:
    with pytest.raises(ValueError, match="overlap"):
        split_text("hello", chunk_size=4, overlap=4)
