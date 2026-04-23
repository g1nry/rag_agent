from pathlib import Path


class DocumentStore:
    def __init__(self, base_dir: Path) -> None:
        self._base_dir = base_dir
        self._base_dir.mkdir(parents=True, exist_ok=True)

    def save(self, filename: str, text: str) -> str:
        safe_name = Path(filename).name or "document.txt"
        target = self._base_dir / safe_name
        target.write_text(text, encoding="utf-8")
        return safe_name

