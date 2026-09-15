FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app/src \
    TRIAGE_MODEL_DIR=/app/models \
    TRIAGE_MODEL_BACKEND=sklearn \
    TRIAGE_ORT_THREADS=1 \
    TRIAGE_LOG_LEVEL=INFO

WORKDIR /app

RUN useradd --system --no-create-home --uid 10001 app

# Ordem deps -> src -> models: retreinar só invalida a última camada.
COPY requirements/api.txt /tmp/api.txt
RUN pip install --no-cache-dir -r /tmp/api.txt && rm /tmp/api.txt

COPY src/ /app/src/
COPY models/ /app/models/

USER app
EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=3s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=2)"]

# 1 worker: o registro do prometheus_client é por processo. Escala por réplicas.
CMD ["uvicorn", "triage.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
