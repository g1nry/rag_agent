import logging
import os
from contextlib import contextmanager
from functools import lru_cache
from typing import Any

logger = logging.getLogger(__name__)


def _is_enabled() -> bool:
    value = os.getenv("LANGFUSE_TRACING_ENABLED", "false").lower().strip()
    return value in {"1", "true", "yes", "on"}


def _clean_dict(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if value is not None}


@lru_cache(maxsize=1)
def get_langfuse_handler():
    """
    Создаёт Langfuse CallbackHandler для LangChain/LangGraph.

    Если Langfuse выключен, SDK не установлен или auth_check не проходит,
    агент не падает, а просто работает без трейсинга.
    """
    if not _is_enabled():
        logger.info("Langfuse tracing is disabled")
        return None

    try:
        from langfuse import get_client
        from langfuse.langchain import CallbackHandler

        langfuse = get_client()

        try:
            if not langfuse.auth_check():
                logger.warning("Langfuse auth_check returned False")
                return None
        except Exception as exc:
            logger.warning("Langfuse auth_check failed: %s", exc)
            return None

        logger.info("Langfuse tracing is enabled")
        return CallbackHandler()

    except Exception as exc:
        logger.warning("Langfuse callback initialization failed: %s", exc)
        return None


def langchain_config(
    *,
    session_id: str = "default",
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Возвращает config для LangGraph/LangChain .ainvoke().
    """
    handler = get_langfuse_handler()
    if handler is None:
        return {}

    tags = tags or []
    metadata = metadata or {}

    langfuse_metadata = {
        **metadata,
        "langfuse_session_id": session_id,
        "langfuse_tags": tags,
    }

    return {
        "callbacks": [handler],
        "metadata": langfuse_metadata,
        "tags": tags,
        "run_name": "rag-agent-chat",
    }


def observe(*args, **kwargs):
    """
    Безопасная обёртка над langfuse.observe.

    Используем для обычного Python-кода, который не проходит через LangChain.
    Если Langfuse выключен или SDK недоступен — возвращаем no-op decorator.
    """
    def identity_decorator(func):
        return func

    if not _is_enabled():
        return identity_decorator

    try:
        from langfuse import observe as langfuse_observe
        return langfuse_observe(*args, **kwargs)
    except Exception as exc:
        logger.warning("Langfuse observe decorator initialization failed: %s", exc)
        return identity_decorator


def update_trace(
    *,
    name: str | None = None,
    session_id: str | None = None,
    user_id: str | None = None,
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
    input: Any | None = None,
    output: Any | None = None,
) -> None:
    """
    Обновляет текущий trace, созданный @observe.
    """
    if not _is_enabled():
        return

    try:
        from langfuse import get_client

        langfuse = get_client()
        langfuse.update_current_trace(
            **_clean_dict(
                {
                    "name": name,
                    "session_id": session_id,
                    "user_id": user_id,
                    "tags": tags,
                    "metadata": metadata,
                    "input": input,
                    "output": output,
                }
            )
        )
    except Exception as exc:
        logger.warning("Langfuse update_current_trace failed: %s", exc)


@contextmanager
def langfuse_observation(
    *,
    name: str,
    as_type: str = "span",
    input: Any | None = None,
    model: str | None = None,
    metadata: dict[str, Any] | None = None,
):
    """
    Создаёт вложенную observation/span/generation внутри текущего trace.

    Пример:
      with langfuse_observation(name="rag-retrieval") as span:
          ...
          span.update(output={...})
    """
    if not _is_enabled():
        yield None
        return

    try:
        from langfuse import get_client

        langfuse = get_client()
        observation_cm = langfuse.start_as_current_observation(
            **_clean_dict(
                {
                    "name": name,
                    "as_type": as_type,
                    "input": input,
                    "model": model,
                    "metadata": metadata,
                }
            )
        )
    except Exception as exc:
        logger.warning("Langfuse observation initialization failed: %s", exc)
        yield None
        return

    with observation_cm as observation:
        yield observation
