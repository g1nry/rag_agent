def _split_long_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    chunks: list[str] = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end == len(text):
            break

        start = end - overlap

    return chunks


def _split_blocks(text: str) -> list[str]:
    blocks: list[str] = []
    current: list[str] = []

    for line in text.splitlines():
        stripped_line = line.strip()

        if not stripped_line:
            if current:
                blocks.append("\n".join(current).strip())
                current = []
            continue

        if stripped_line.startswith("#") and current:
            blocks.append("\n".join(current).strip())
            current = []

        current.append(stripped_line)

    if current:
        blocks.append("\n".join(current).strip())

    return [block for block in blocks if block]


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
    current = ""

    for block in _split_blocks(stripped_text):
        if len(block) > chunk_size:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(_split_long_text(block, chunk_size, overlap))
            continue

        candidate = f"{current}\n\n{block}" if current else block
        if len(candidate) <= chunk_size:
            current = candidate
            continue

        if current:
            chunks.append(current)
        current = block

    if current:
        chunks.append(current)

    return chunks
