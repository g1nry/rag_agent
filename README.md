# RAG Agent Backend

Независимый backend-сервис агента и RAG для LLM-системы. Проект запускается отдельно от инфраструктурного стенда, работает через Ollama как через внешний model backend и предоставляет HTTP API для чата, retrieval и загрузки документов.

## Что внутри

- FastAPI backend
- отдельный интеграционный слой `integrations/ollama`
- базовая orchestration-логика агента
- простой RAG pipeline: ingestion -> chunking -> embeddings -> retrieval
- минимальный встроенный UI для локального тестирования
- единый `config.toml` для настройки сервиса

## Ограничения MVP

- сейчас поддерживается только Ollama
- для чата и embeddings нужны доступные модели в Ollama
- сервис не зависит от Ansible и не содержит инфраструктурной логики
- встроенный UI предназначен только для локальной проверки backend-логики

## Структура

```text
.
├── src/rag_agent/
│   ├── api/                        HTTP API и роуты
│   ├── agent/                      orchestration логика агента
│   ├── core/                       конфиг и общие утилиты
│   ├── domain/                     pydantic-схемы
│   ├── integrations/ollama/        клиент работы с Ollama
│   ├── rag/                        chunking и similarity
│   ├── services/                   ingestion, retrieval, llm
│   ├── storage/                    локальные файловые и индексные хранилища
│   ├── __main__.py                 entrypoint запуска сервиса
│   └── main.py                     FastAPI-приложение
├── frontend/                       минимальный тестовый UI
├── data/                           локальные документы и индексы
├── tests/                          тесты
├── config.toml                     основной конфиг приложения
├── .env.example                    пример env-переопределений
├── pyproject.toml                  зависимости и настройки проекта
├── Dockerfile                      контейнеризация backend-сервиса
├── docker-compose.yml              локальный запуск в контейнере
└── README.md
```

## Конфигурация

Главный конфиг сервиса хранится в `config.toml`. Через него настраиваются:

- параметры HTTP-сервера
- адрес и модели Ollama
- таймауты клиентского обращения к Ollama
- пути к данным, документам и индексу
- параметры chunking и retrieval
- включение или отключение встроенного UI

Текущий пример:

```toml
[app]
name = "rag-agent"
host = "0.0.0.0"
port = 8000

[ollama]
base_url = "http://localhost:11434"
chat_model = "llama3.2:1b"
embedding_model = "nomic-embed-text"
timeout = 60.0

[data]
dir = "./data"
documents_dir = "./data/documents"
index_path = "./data/indexes/vector_index.json"

[rag]
max_chunk_size = 800
chunk_overlap = 120
default_top_k = 4
min_retrieval_score = 0.2

[documents]
max_upload_size_bytes = 1048576
allowed_extensions = [".txt", ".md"]

[ui]
enabled = true
```

Если нужно использовать другой конфиг, можно указать путь через переменную окружения:

```bash
RAG_AGENT_CONFIG=/path/to/config.toml
```

`config.toml` является основным способом настройки. `.env` не обязателен и нужен только как слой env-overrides при интеграции или деплое.

Примеры env-overrides:

```bash
export RAG_AGENT_CONFIG=/srv/rag-agent/config.toml
export APP_PORT=8080
export OLLAMA_BASE_URL=http://ollama.internal:11434
export UI_ENABLED=false
```

## Локальный запуск

### Что нужно установить

- Python 3.11+
- `python3-venv`
- `python3-pip`
- `build-essential`
- `python3-dev`
- `curl`
- Ollama

Пример для Ubuntu/Debian:

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip build-essential python3-dev curl
```

После установки Ollama нужно убедиться, что сервис доступен и что загружены модели:

```bash
ollama pull llama3.2:1b
ollama pull nomic-embed-text
```

### Быстрый старт

1. Создать окружение и установить зависимости:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

2. При необходимости отредактировать `config.toml`.

3. Запустить backend:

```bash
python3 -m rag_agent --reload
```

4. Открыть UI:

`http://localhost:8000/`

Если в `config.toml` изменен `app.port`, открывать нужно новый порт.

## Запуск в контейнере

### Через Docker Compose

```bash
docker compose up --build
```

Текущий `docker-compose.yml`:

- монтирует `./config.toml` в `/app/config.toml`
- монтирует `./data` в `/app/data`
- запускает сервис через `python -m rag_agent`

Это значит, что документы и индекс сохраняются между перезапусками контейнера.

### Через Docker напрямую

Сборка:

```bash
docker build -t rag-agent .
```

Запуск:

```bash
docker run --rm \
  -p 8000:8000 \
  -v "$(pwd)/config.toml:/app/config.toml:ro" \
  -v "$(pwd)/data:/app/data" \
  rag-agent
```

### Важное замечание про Ollama в контейнере

Если backend работает в контейнере, `ollama.base_url = "http://localhost:11434"` подходит только в том случае, если Ollama доступна именно изнутри того же сетевого пространства. Если Ollama запущена снаружи контейнера или как отдельный сервис, `base_url` нужно изменить под реальный адрес.

Например:

```toml
[ollama]
base_url = "http://host.docker.internal:11434"
chat_model = "llama3.2:1b"
embedding_model = "nomic-embed-text"
timeout = 60.0
```

или:

```toml
[ollama]
base_url = "http://ollama:11434"
chat_model = "llama3.2:1b"
embedding_model = "nomic-embed-text"
timeout = 60.0
```

Конкретное значение зависит от того, как именно организовано окружение.

## Запуск тестов

Все тесты:

```bash
python3 -m pytest
```

Отдельный тест:

```bash
python3 -m pytest tests/test_retrieval_service.py -v
```

## Основные endpoint'ы

- `GET /health` — проверка доступности сервиса
- `POST /api/v1/chat` — генерация ответа через agent-логику
- `POST /api/v1/rag/search` — retrieval без генерации
- `POST /api/v1/documents/upload` — загрузка документа в индекс

## Ограничения загрузки документов

Загрузка документов через `POST /api/v1/documents/upload` теперь валидируется до индексации.

По умолчанию:

- разрешены только `.txt` и `.md`
- максимальный размер файла — `1048576` байт
- файл должен быть UTF-8 encoded text
- пустой файл не принимается

Настройки находятся в секции `[documents]`:

```toml
[documents]
max_upload_size_bytes = 1048576
allowed_extensions = [".txt", ".md"]
```

Если проверка не проходит, backend возвращает структурированную ошибку.

Основные сценарии:

- `400` — пустой файл или неверная кодировка
- `413` — файл слишком большой
- `415` — неподдерживаемый тип документа

## Ошибки Ollama

Если backend не может корректно обратиться к Ollama, API теперь возвращает структурированные ошибки вместо общего внутреннего сбоя.

Основные сценарии:

- `503` — Ollama недоступна или до нее не удалось подключиться
- `504` — запрос к Ollama превысил таймаут
- `502` — Ollama вернула некорректный или неожиданный ответ

Пример ответа:

```json
{
  "detail": "Failed to connect to Ollama at http://localhost:11434.",
  "error_code": "ollama_unavailable"
}
```

## Надежность индекса

Локальный JSON-индекс теперь записывается атомарно: сначала данные пишутся во временный файл, затем он заменяет основной `vector_index.json`.

Это снижает риск повреждения индекса при сбоях во время записи.

Если индекс на диске поврежден или имеет неверный формат, backend возвращает структурированную storage-ошибку вместо неясного падения.

Основные сценарии:

- `500` — индекс не удалось прочитать
- `500` — индекс не удалось записать

Пример ответа:

```json
{
  "detail": "Vector index contains invalid JSON.",
  "error_code": "index_load_error"
}
```

## Типовой сценарий работы

1. Пользователь отправляет запрос из UI или внешнего веба.
2. Backend при необходимости выполняет retrieval по загруженным документам.
3. Backend передает запрос и релевантный контекст в Ollama.
4. Backend возвращает ответ и найденные фрагменты контекста.

## Формат retrieval-контекста

Для endpoint'ов, связанных с RAG, backend возвращает контекст как список объектов:

```json
{
  "text": "Фрагмент текста",
  "source": "README.md",
  "chunk_id": "README.md:0"
}
```

Это позволяет внешнему клиенту показывать не только текст, но и источник чанка.

Пример запроса к чату:

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "О чем этот проект?",
    "use_rag": true,
    "top_k": 3
  }'
```

Пример ответа:

```json
{
  "answer": "Краткий ответ модели",
  "contexts": [
    {
      "text": "Фрагмент релевантного контекста 1",
      "source": "README.md",
      "chunk_id": "README.md:0"
    },
    {
      "text": "Фрагмент релевантного контекста 2",
      "source": "README.md",
      "chunk_id": "README.md:1"
    }
  ]
}
```

## Интеграция с внешним вебом

Встроенный `frontend/index.html` является только минимальным тестовым UI. Он не должен определять архитектуру системы и не обязателен для production-интеграции.

Если внешний веб уже существует, обычно достаточно:

1. Поднять этот backend как отдельный HTTP API-сервис.
2. Указать во внешнем вебе адрес backend.
3. Вызывать нужные endpoint'ы:
   - `POST /api/v1/chat`
   - `POST /api/v1/rag/search`
   - `POST /api/v1/documents/upload`
   - `GET /health`
4. Корректно обрабатывать поле `contexts` как список объектов, а не строк.

Упрощенная схема:

`external web -> this backend -> Ollama`

Если нужно отключить встроенный UI, это можно сделать через конфиг:

```toml
[ui]
enabled = false
```

## Что обычно меняют под свой проект

### Ollama

Меняют секцию `[ollama]` в `config.toml`:

- `base_url`
- `chat_model`
- `embedding_model`
- `timeout`

### Пути данных

Меняют секцию `[data]`:

- `dir`
- `documents_dir`
- `index_path`

### Параметры retrieval

Меняют секцию `[rag]`:

- `max_chunk_size`
- `chunk_overlap`
- `default_top_k`
- `min_retrieval_score`

Если retrieval возвращает слишком много нерелевантного контекста, имеет смысл увеличить `min_retrieval_score`. Если retrieval слишком часто ничего не находит, порог можно уменьшить.

### HTTP-параметры сервиса

Меняют секцию `[app]`:

- `host`
- `port`
- `name`

## Типичные проблемы

### Не создается виртуальное окружение

Если команда

```bash
python3 -m venv .venv
```

не работает, чаще всего не установлен пакет `python3-venv`.

### Backend запускается, но чат не отвечает

Чаще всего это означает, что Ollama не запущена, недоступна по сети или в `config.toml` указан неверный `ollama.base_url`.

Что проверить:

- что Ollama действительно запущена
- что адрес в `config.toml` совпадает с реальным адресом сервиса
- что backend может достучаться до Ollama по HTTP API
- какой код ошибки вернул backend: `503`, `504` или `502`

### Ollama доступна, но генерация или embeddings не работают

Обычно причина в том, что нужные модели не загружены.

Что проверить:

- доступна ли chat-модель, например `llama3.2:1b`
- доступна ли embedding-модель, например `nomic-embed-text`

### Retrieval ничего не находит

Возможные причины:

- документы еще не были загружены в индекс
- `min_retrieval_score` слишком высокий
- запрос и загруженные документы действительно слабо связаны по смыслу

### Сервис не стартует из-за занятого порта

Если порт занят, измените `app.port` в `config.toml`, например:

```toml
[app]
port = 8001
```

### После запуска не открывается локальный UI

Что проверить:

- что backend действительно запущен
- что открыт правильный адрес и порт
- что в конфиге не выключен UI

Пример:

```toml
[ui]
enabled = true
```
