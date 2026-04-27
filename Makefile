.PHONY: install run test

PYTHON ?= python3

install:
	$(PYTHON) -m pip install -e .

run:
	$(PYTHON) -m rag_agent --reload

test:
	$(PYTHON) -m pytest
