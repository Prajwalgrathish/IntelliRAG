PYTHON ?= python

.PHONY: install install-dev train ingest index serve evaluate test lint docker-build docker-run clean

install:
	$(PYTHON) -m pip install -r requirements.txt

install-dev:
	$(PYTHON) -m pip install -r requirements-dev.txt

train:
	$(PYTHON) scripts/train_classifier.py --data data/training/intent_queries.csv

ingest:
	$(PYTHON) scripts/ingest_documents.py --input-dir data/documents

index:
	$(PYTHON) scripts/build_index.py --rebuild

serve:
	uvicorn app.main:app --host 0.0.0.0 --port 8000

evaluate:
	$(PYTHON) scripts/evaluate_rag.py --eval-file data/evaluation/eval_set.example.jsonl

test:
	$(PYTHON) -m pytest

lint:
	ruff check app src scripts tests

docker-build:
	docker build -t intellirag:latest .

docker-run:
	docker compose up --build

clean:
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov
