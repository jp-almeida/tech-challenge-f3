FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app/src \
    TRIAGE_MODEL_DIR=/app/models \
    TRIAGE_MODEL_BACKEND=onnx \
    TRIAGE_ORT_THREADS=1 \
    TRIAGE_LOG_LEVEL=INFO

WORKDIR /app

# O operador StringNormalizer do ONNX Runtime abre o locale en_US.UTF-8, que não existe
# na imagem slim: sem isso a sessão ONNX falha em "Failed to construct locale".
RUN apt-get update \
    && apt-get install -y --no-install-recommends locales \
    && sed -i '/^# *en_US.UTF-8 UTF-8/s/^# *//' /etc/locale.gen \
    && locale-gen \
    && rm -rf /var/lib/apt/lists/*
ENV LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8

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
