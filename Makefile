.PHONY: install run test

install:
	python -m pip install -e .

run:
	uvicorn rag_agent.main:app --reload

test:
	pytest

