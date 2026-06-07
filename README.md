# RAG RedTeam Agent v0.3.0

**Полноценный агент с RAG + опасными инструментами** для исследования уязвимостей LLM.

---

## 1. Что это такое и зачем нужно

Это **бэкенд-сервис**, который запускается отдельно от твоего основного приложения. Он предоставляет HTTP API, через которое можно общаться с умным агентом на базе большой языковой модели (LLM).

### Зачем это нужно?

Обычные LLM (типа ChatGPT) могут только **отвечать на вопросы**.  
Наш агент может **действовать**:
- Выполнять команды в твоей системе
- Читать и писать файлы
- Менять права доступа
- Искать информацию в твоих документах

Это позволяет **исследовать уязвимости** LLM — например, проверять, может ли модель выполнить опасную команду, если её правильно попросить.

**Простыми словами:** это инструмент для тестирования, насколько "умная" и "послушная" модель, и какие действия она готова выполнить.

---

## 2. Что умеет агент

### Основные возможности:

1. **Обычный чат** — отвечает на вопросы, как обычный ChatGPT
2. **RAG (поиск по документам)** — может искать информацию в загруженных файлах
3. **Выполнение команд** — может запускать любые команды в терминале (`ls`, `cat`, `whoami` и т.д.)
4. **Работа с файлами**:
   - Читать файлы (`file_read`)
   - Писать в файлы (`file_write`)
   - Удалять файлы и папки (`file_delete`)
   - Создавать папки (`mkdir`)
   - Менять права доступа (`chmod`)

### Пример работы:

**Ты пишешь:**
> "выполни команду ls -la"

**Агент:**
1. Понимает, что нужно выполнить команду
2. Вызывает инструмент `shell_execute`
3. Получает реальный результат
4. Возвращает тебе красивый ответ

---

## 3. Требования к системе

### Минимальные:
- **Python 3.11+**
- **8 ГБ RAM** (для модели 7B)
- **Ollama** (локально или по сети)
- **Модель Qwen2.5:7b** (рекомендуется)

### Рекомендуемые:
- **16+ ГБ RAM**
- **GPU** (NVIDIA с CUDA) — сильно ускоряет работу
- **Быстрый SSD**

### Поддерживаемые модели (Ollama):

| Модель          | Качество | Скорость | Рекомендация      |
|-----------------|----------|----------|-------------------|
| qwen2.5:7b      | Отличное | Средняя  | **Лучший выбор**  |
| llama3.1:8b     | Отличное | Средняя  | Хорошая альтернатива |
| llama3.2:3b     | Среднее  | Быстрая  | Для слабых ПК     |
| phi3:mini       | Среднее  | Быстрая  | Для тестов        |

---

## 4. Быстрый старт (5 минут)

```bash
# 1. Клонируй репозиторий
git clone https://github.com/g1nry/rag_agent.git
cd rag_agent

# 2. Создай виртуальное окружение
python3 -m venv .venv
source .venv/bin/activate

# 3. Установи зависимости
pip install -e .

# 4. Запусти Ollama (в отдельном терминале)
ollama serve

# 5. Скачай модель (в отдельном терминале)
ollama pull qwen2.5:7b

# 6. Запусти бэкенд
python -m rag_agent
```

Готово! Открой http://localhost:8000 в браузере.

---

## 5. Подробная установка (для новичков)

### Шаг 1: Установка Python

**Linux (Ubuntu/Debian):**

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip build-essential python3-dev curl
```

**Проверка:**

```bash
python3 --version
# Должно быть: Python 3.11.0 или выше
```

### Шаг 2: Виртуальное окружение

```bash
cd ~/workspaces/rag_agent
python3 -m venv .venv
source .venv/bin/activate
```

**Важно:** Каждый раз, когда ты работаешь с проектом, нужно активировать окружение:

```bash
source .venv/bin/activate
```

### Шаг 3: Установка зависимостей

```bash
pip install --upgrade pip
pip install -e .
```

Это установит все нужные библиотеки (FastAPI, LangGraph, LangChain, httpx и т.д.).

### Шаг 4: Ollama (самое важное!)

1. Перейди на https://ollama.com
2. Скачай и установи Ollama для своей ОС
3. Запусти:

```bash
ollama serve
```

4. В **новом терминале** скачай модель:

```bash
ollama pull qwen2.5:7b
```

**Это займёт время** (модель весит ~4-5 ГБ).

---

## 6. Запуск сервиса

### Вариант 1: Простой запуск

```bash
python -m rag_agent
```

### Вариант 2: С автоперезагрузкой (удобно при разработке)

```bash
uvicorn src.rag_agent.main:app --reload --port 8000
```

### Вариант 3: Через Docker

```bash
docker compose up --build
```

---

## 7. Как пользоваться

### Через браузер (просто для теста)

Открой http://localhost:8000 — там будет простая страница с описанием.

### Через curl (основной способ)

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "твой вопрос или команда здесь",
    "thread_id": "default"
  }'
```

Поля JSON-запроса:
- `message` — текст вопроса или команды.
- `thread_id` — идентификатор диалога, по умолчанию `"default"`.

Для вопросов по загруженным документам используй `/api/v1/rag/chat`. Там дополнительно есть `top_k` — сколько фрагментов из RAG-индекса взять в контекст. Этот endpoint отвечает напрямую по RAG-контексту и не запускает агента с инструментами.

### Через Python (для интеграции)

```python
import requests

response = requests.post(
    "http://localhost:8000/api/agent/chat",
    json={"message": "выполни команду whoami"}
)

print(response.json()["answer"])
```

---

## 8. Основные эндпоинты API

| Метод | Эндпоинт                        | Назначение                                      |
|-------|---------------------------------|-------------------------------------------------|
| POST  | `/api/v1/documents/upload`      | Загрузка и индексация документа                 |
| GET   | `/api/v1/documents`             | Список всех загруженных документов              |
| DELETE| `/api/v1/documents/{filename}`  | Удаление документа из индекса                   |
| POST  | `/api/v1/chat`                  | Простой RAG (без опасных инструментов)          |
| POST  | `/api/v1/rag/search`            | Только поиск релевантных чанков                 |
| POST  | `/api/agent/chat`               | Полноценный ReAct-агент с инструментами         |
| GET   | `/health`                       | Проверка состояния                              |

---

## 9. Хранилище

- **Vector Store**: SQLite (`VectorStore`)
- Поддержка миграции со старого JSON-индекса
- Удобные методы: `get_documents()`, `delete_document()`, `replace_document()`

---

## 10. Примеры запросов (curl)

### Пример 1: Обычный вопрос

```bash
curl -X POST http://localhost:8000/api/v1/rag/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "О чем этот проект? Ответь на основе README.md",
    "thread_id": "default",
    "top_k": 3
  }'
```

### Пример 2: Загрузка документа в RAG

```bash
curl -X POST http://localhost:8000/api/v1/documents/upload \
  -F "file=@./README.md"

Ответ:

```json
{
  "document_id": "2e9f7a24-9c73-42a7-9c1e-7ef9e20c2a1a",
  "filename": "README.md",
  "status": "queued"
}
```

Проверь статус индексации перед вопросами по документу:

```bash
curl http://localhost:8000/api/v1/documents/2e9f7a24-9c73-42a7-9c1e-7ef9e20c2a1a/status
```

Готовый документ выглядит так:

```json
{
  "document_id": "2e9f7a24-9c73-42a7-9c1e-7ef9e20c2a1a",
  "filename": "README.md",
  "status": "indexed",
  "chunks_indexed": 20,
  "error": null,
  "message": null
}
```

`rag_search` — это внутренний инструмент `/api/agent/chat` для поиска по документам, загруженным через `/chat/v1/documents/upload`. Обычный RAG endpoint `/api/v1/rag/chat` не вызывает agent tools и не должен упоминать `rag_search`; он сначала достает контекст из индекса, затем напрямую просит LLM ответить по найденным фрагментам.

### Пример 3: Выполнение команды

```bash
curl -X POST http://localhost:8000/api/agent/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "выполни команду ls -la",
    "thread_id": "default"
  }'
```

### Пример 4: Создание файла

```bash
curl -X POST http://localhost:8000/api/agent/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "создай файл test.txt с текстом \"Привет от агента\"",
    "thread_id": "default"
  }'
```

### Пример 5: Чтение файла

```bash
curl -X POST http://localhost:8000/api/agent/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "прочитай файл /etc/passwd",
    "thread_id": "default"
  }'
```

### Пример 6: Изменение прав

```bash
curl -X POST http://localhost:8000/api/agent/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "измени права файла test.txt на 777",
    "thread_id": "default"
  }'
```

---

## 11. Структура проекта

```
rag_agent/
├── src/
│   └── rag_agent/
│       ├── agents/           # Главный агент (RedTeamAgent)
│       │   └── main_agent.py
│       ├── api/              # FastAPI роуты
│       │   └── router.py
│       ├── core/             # Конфиг
│       │   └── config.py
│       ├── security/         # HITL и разрешения
│       │   ├── hitl.py
│       │   └── permission.py
│       ├── services/         # RAG и LLM
│       |   ├── document_ingestion_service.py
│       │   ├── retrieval_service.py
│       │   └── llm_service.py
│       ├── storage/          # Хранилище индекса
│       │   ├── vector_store.py
│       |   └── document_store.py
│       ├── tools/            # Опасные инструменты
│       │   ├── base.py
│       │   ├── dangerous_tools.py
│       │   ├── rag_tool.py
│       │   └── registry.py
│       ├── main.py           # FastAPI приложение
│       └── __main__.py       # CLI entrypoint
├── data/                     # Документы и индекс
├── frontend/                 # Минимальный UI
├── config.toml               # Конфиг
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
└── README.md
```

---

## 12. Конфигурация (config.toml)

### Основные секции:

```toml
[app]
name = "rag-redteam-agent"
host = "0.0.0.0"
port = 8000

[ollama]
base_url = "http://localhost:11434"
chat_model = "qwen2.5:7b"
embedding_model = "nomic-embed-text"
timeout = 120.0

[data]
dir = "./data"
documents_dir = "./data/documents"
index_path = "./data/indexes/vector_index.json"

[rag]
max_chunk_size = 800
chunk_overlap = 120
default_top_k = 4
min_retrieval_score = 0.2
```

### Важные параметры:

| Параметр                    | Что делает                              | Рекомендация          |
|----------------------------|-----------------------------------------|-----------------------|
| `chat_model`               | Какая модель отвечает                   | qwen2.5:7b            |
| `timeout`                  | Сколько ждать ответа от модели          | 120 (для 7B)          |
| `min_retrieval_score`      | Минимальная релевантность документа     | 0.2                   |
| `default_top_k`            | Сколько документов брать                | 4                     |

---

## 13. Как работает агент (просто)

1. **Ты пишешь сообщение**
2. **Агент думает** (LLM решает, нужно ли использовать инструменты)
3. **Если нужно** — вызывает инструмент (например, `shell_execute`)
4. **Получает результат** от инструмента
5. **Думает снова** (учитывая результат)
6. **Выдаёт финальный ответ**

Это называется **ReAct** (Reason + Act).

---

## 14. Опасные инструменты (подробно)

### `shell_execute`
Выполняет любую команду в терминале.

**Пример:**
```json
{"command": "ls -la"}
{"command": "cat /etc/passwd"}
{"command": "whoami"}
```

**Опасность:** Высокая. Может выполнить `rm -rf /` или скачать вредоносный код.

### `file_read`
Читает содержимое файла.

**Пример:**
```json
{"filepath": "/etc/passwd"}
```

### `file_write`
Записывает текст в файл (перезаписывает!).

**Пример:**
```json
{"filepath": "test.txt", "content": "Привет"}
```

### `file_delete`
Удаляет файл или папку.

**Пример:**
```json
{"filepath": "test.txt"}
```

### `mkdir`
Создаёт директорию.

**Пример:**
```json
{"path": "new_folder/subfolder"}
```

### `chmod`
Меняет права доступа.

**Пример:**
```json
{"filepath": "test.txt", "mode": "777"}
```

---

## 15. Возможные ошибки и как их исправить

### Ошибка 1: Модель отвечает на китайском

**Причина:** Модель плохо понимает русский или `temperature` слишком высокая.

**Решение:**
- Используй `qwen2.5:7b`
- В `main_agent.py` поставь `temperature=0.3`

### Ошибка 2: `Connection refused` (Ollama)

**Причина:** Ollama не запущена или неправильный `base_url`

**Решение:**
```bash
ollama serve
# Проверь base_url в config.toml
```

### Ошибка 3: Агент "не видит" созданные файлы

**Причина:** Слабая модель галлюцинирует.

**Решение:** Используй `qwen2.5:7b` или `llama3.1:8b`

---

## 16. Docker (контейнеризация)

### docker-compose.yml (актуальный)

```yaml
version: "3.9"

services:
  backend:
    build: .
    container_name: rag-redteam-agent
    ports:
      - "8000:8000"
    volumes:
      - ./config.toml:/app/config.toml:ro
      - ./data:/app/data
    environment:
      - OLLAMA_BASE_URL=http://host.docker.internal:11434
    restart: unless-stopped
```

### Запуск

```bash
docker compose up --build
```

---

## 17. Часто задаваемые вопросы

**Q: Можно ли использовать модель из OpenAI вместо Ollama?**  
A: Пока нет. Поддерживается только Ollama.

**Q: Безопасно ли запускать этот агент?**  
A: Нет. Агент может выполнять опасные команды. Используй только в изолированном окружении.

**Q: Можно ли добавить свои инструменты?**  
A: Да. Создай новый класс в `tools/dangerous_tools.py` и добавь его в `create_dangerous_tools()`.

**Q: Как сделать так, чтобы агент спрашивал подтверждение перед опасными действиями?**  
A: Это уже реализовано в `security/hitl.py`. Нужно подключить к фронтенду.

---

## Chunking Notes

Документы режутся на чанки с учетом структуры текста: Markdown-заголовки, абзацы и короткие блоки сохраняются вместе, а слишком длинные блоки режутся по `max_chunk_size` с overlap из `chunk_overlap`.

---

## 18. Roadmap (что будет дальше)

- [x] ReAct-агент с 6 инструментами
- [x] RAG + агент в одном
- [ ] Полноценный HITL с подтверждением через API
- [ ] Мультиагентность (несколько агентов)
- [ ] Логирование всех действий
- [ ] Метрики и дашборд
