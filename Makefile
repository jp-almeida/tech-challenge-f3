PY := .venv/bin/python
PIP := .venv/bin/pip
export PYTHONPATH := src
BACKEND ?= sklearn

.PHONY: setup lint format test data train api docker-build up down up-airflow dag-test test-dag load-test bench-models bench-api clean

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

# Executa a DAG inteira sem scheduler (logs no terminal).
dag-test:
	docker compose --profile airflow run --rm --entrypoint bash airflow -c \
		"airflow db migrate >/dev/null && airflow dags test triage_training"

# Testes de integridade da DAG dentro da imagem do Airflow (no .venv eles são pulados).
test-dag:
	docker compose --profile airflow build airflow
	docker run --rm --entrypoint bash -e PYTHONPATH=/opt/airflow/src \
		-v "$(CURDIR)/src:/opt/airflow/src" -v "$(CURDIR):/opt/airflow/project" \
		triage-airflow:local -c "pip install -q --user pytest && airflow db migrate >/dev/null && \
		cd /opt/airflow/project && python -m pytest -p no:cacheprovider -o addopts='' tests/test_dag.py"

load-test:
	$(PY) scripts/load_test.py --duration 120

bench-models:
	$(PY) scripts/benchmark_models.py

bench-api:
	$(PY) scripts/benchmark_api.py --label $(BACKEND)

clean:
	rm -rf .pytest_cache .ruff_cache .coverage data/processed/*.csv models/candidates/*
