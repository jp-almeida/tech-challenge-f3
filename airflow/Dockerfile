# Build context = raiz do repositório (para enxergar requirements/).
FROM apache/airflow:3.3.1-python3.12

# A imagem já traz scikit-learn 1.9 e pandas 3.0. Os pins de train.txt precisam vencer:
# o modelo treinado aqui é carregado pela API, que usa scikit-learn 1.8.0.
# Fixar apache-airflow na mesma instalação impede o pip de atualizá-lo ou rebaixá-lo.
# A task export_onnx valida a paridade abrindo uma sessão do ONNX Runtime, cujo operador
# StringNormalizer exige o locale en_US.UTF-8 (ausente na imagem base).
USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends locales \
    && sed -i '/^# *en_US.UTF-8 UTF-8/s/^# *//' /etc/locale.gen \
    && locale-gen \
    && rm -rf /var/lib/apt/lists/*
ENV LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8
USER airflow

COPY requirements/train.txt /tmp/train.txt
RUN pip install --no-cache-dir "apache-airflow==3.3.1" -r /tmp/train.txt
