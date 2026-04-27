from pathlib import Path

from rag_agent.core.config import load_settings


def test_load_settings_from_toml(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
[app]
name = "custom-agent"
host = "127.0.0.1"
port = 9000

[ollama]
base_url = "http://example.com:11434"

[data]
dir = "./runtime-data"

[rag]
default_top_k = 7

[ui]
enabled = false
""".strip(),
        encoding="utf-8",
    )

    settings = load_settings(config_path)

    assert settings.app_name == "custom-agent"
    assert settings.app_host == "127.0.0.1"
    assert settings.app_port == 9000
    assert str(settings.ollama_base_url) == "http://example.com:11434/"
    assert settings.data_dir == (tmp_path / "runtime-data").resolve()
    assert settings.documents_dir == (tmp_path / "runtime-data" / "documents").resolve()
    assert settings.index_path == (tmp_path / "runtime-data" / "indexes" / "vector_index.json").resolve()
    assert settings.default_top_k == 7
    assert settings.ui_enabled is False


def test_env_overrides_take_priority(tmp_path: Path, monkeypatch) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
[app]
port = 8000

[ollama]
base_url = "http://localhost:11434"
chat_model = "llama3.2:1b"
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("APP_PORT", "8100")
    monkeypatch.setenv("OLLAMA_CHAT_MODEL", "llama3.1:8b")

    settings = load_settings(config_path)

    assert settings.app_port == 8100
    assert settings.ollama_chat_model == "llama3.1:8b"
