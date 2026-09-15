PY := .venv/bin/python
PIP := .venv/bin/pip
export PYTHONPATH := src
BACKEND ?= sklearn

.PHONY: setup lint format test data train api docker-build up down up-airflow load-test bench-models bench-api clean

setup:
	python3.12 -m venv .venv || uv venv --seed --python 3.12 .venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements/dev.txt

lint:
	.venv/bin/ruff check .
	.venv/bin/ruff format --check .

format:
	.venv/bin/ruff format .
	.venv/bin/ruff check --fix .

test:
	.venv/bin/pytest --cov=src/triage --cov-report=term-missing

data:
	$(PY) scripts/download_data.py

train:
	$(PY) -m triage.pipeline run-all

api:
	.venv/bin/uvicorn triage.api.main:app --reload

docker-build:
	docker build -t triage-api:local .

up:
	docker compose up -d --build

down:
	docker compose down

up-airflow:
	docker compose --profile airflow up -d --build

load-test:
	$(PY) scripts/load_test.py --duration 120

bench-models:
	$(PY) scripts/benchmark_models.py

bench-api:
	$(PY) scripts/benchmark_api.py --label $(BACKEND)

clean:
	rm -rf .pytest_cache .ruff_cache .coverage data/processed/*.csv models/candidates/*
