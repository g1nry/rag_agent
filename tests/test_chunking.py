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


def test_split_text_keeps_markdown_blocks_together() -> None:
    text = """# Title
Intro paragraph.

## Setup
Install dependencies.

Run the service."""

    assert split_text(text, chunk_size=40, overlap=5) == [
        "# Title\nIntro paragraph.",
        "## Setup\nInstall dependencies.",
        "Run the service.",
    ]


def test_split_text_packs_small_paragraphs() -> None:
    text = """First short paragraph.

Second short paragraph."""

    assert split_text(text, chunk_size=80, overlap=5) == [
        "First short paragraph.\n\nSecond short paragraph."
    ]
