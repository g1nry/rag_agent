from rag_agent.rag.chunking import split_text


def test_split_text_returns_chunks() -> None:
    text = "a" * 1000
    chunks = split_text(text, chunk_size=300, overlap=50)

    assert len(chunks) >= 3
    assert all(chunks)
