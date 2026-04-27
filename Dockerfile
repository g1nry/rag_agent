FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml README.md ./
COPY config.toml ./config.toml
COPY src ./src
COPY frontend ./frontend

RUN pip install --no-cache-dir -e .

CMD ["python", "-m", "rag_agent"]
