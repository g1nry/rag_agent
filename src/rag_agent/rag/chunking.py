def split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    normalized = text.replace("\r\n", "\n").strip()
    if not normalized:
        return []
    
    paragraphs = [part.strip() for part in normalized.split("\n\n") if part.strip()]
    if not paragraphs:
        return []

    chunks: list[str] = []
    current_chunk = ""
    
    for paragraph in paragraphs:
        candidate = paragraph if not current_chunk else f"{current_chunk}\n\n{paragraph}"

        if len(candidate) <= chunk_size:
            current_chunk = candidate
            continue
        
        if current_chunk:
            chunks.append(current_chunk)

        if len(paragraph) <= chunk_size:
            current_chunk = paragraph
            continue

        start = 0
        while start < len(paragraph):
            end = min(len(paragraph), start + chunk_size)
            piece = paragraph[start:end].strip()
            if piece:
                chunks.append(piece)
            if end >= len(paragraph):
                current_chunk = ""
                break
            start = max(0, end - overlap)

    if current_chunk:
        chunks.append(current_chunk)

    return chunks

