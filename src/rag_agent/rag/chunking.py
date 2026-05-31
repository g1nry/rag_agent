def split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")

    if overlap < 0:
        raise ValueError("overlap must be greater than or equal to 0")

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    stripped_text = text.strip()
    if not stripped_text:
        return []

    chunks: list[str] = []
    start = 0

    while start < len(stripped_text):
        end = min(start + chunk_size, len(stripped_text))
        chunk = stripped_text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end == len(stripped_text):
            break

        start = end - overlap

    return chunks
