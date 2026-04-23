FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
COPY frontend ./frontend

RUN pip install --no-cache-dir -e .

CMD ["uvicorn", "rag_agent.main:app", "--host", "0.0.0.0", "--port", "8000"]

