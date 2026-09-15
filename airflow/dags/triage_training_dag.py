"""DAG de treino/retreino do classificador de triagem.

ingest_data -> train_model -> evaluate_model -> quality_gate -> promote_model

Toda a lógica vive em `triage.pipeline` (testada sem Airflow); aqui só há orquestração.
Entre tasks trafegam apenas caminhos e métricas pequenas via XCom, nunca DataFrames.
Imports pesados (sklearn, pandas) ficam dentro das tasks para o parsing da DAG ser rápido.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta

from airflow.sdk import dag, task

from triage.config import DEFAULT_PARAMS, Settings


def _candidate_dir(run_id: str) -> str:
    safe_run_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", run_id)
    return str(Settings().model_dir / "candidates" / safe_run_id)


@dag(
    dag_id="triage_training",
    # Disparo manual. Em produção: "@weekly" ou acionado por alerta de drift.
    schedule=None,
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["triage", "training"],
    default_args={"retries": 1, "retry_delay": timedelta(minutes=1)},
    params={
        "max_features": DEFAULT_PARAMS["max_features"],
        "C": DEFAULT_PARAMS["C"],
        "min_f1": Settings().quality_gate_f1,
    },
)
def triage_training():
    @task
    def ingest_data() -> dict:
        from triage import pipeline

        return pipeline.ingest(Settings().data_dir)

    @task
    def train_model(ingested: dict, run_id=None, params=None) -> dict:
        from triage import pipeline

        overrides = {"max_features": int(params["max_features"]), "C": float(params["C"])}
        result = pipeline.train(ingested["train_path"], _candidate_dir(run_id), overrides)
        return {**result, "candidate_dir": _candidate_dir(run_id)}

    @task
    def evaluate_model(trained: dict, ingested: dict) -> dict:
        from triage import pipeline

        return pipeline.evaluate(
            trained["model_path"], ingested["test_path"], Settings().reports_dir
        )

    # Sem retry: reexecutar não muda o F1 do candidato.
    @task(retries=0)
    def quality_gate(metrics: dict, params=None) -> None:
        from triage import pipeline

        pipeline.quality_gate(metrics, float(params["min_f1"]))

    @task
    def promote_model(trained: dict, metrics: dict, ingested: dict, run_id=None) -> dict:
        from triage import pipeline

        return pipeline.promote(
            trained["candidate_dir"],
            Settings().model_dir,
            metrics,
            extra={
                "n_train": ingested["n_train"],
                "n_test": ingested["n_test"],
                "trained_by": "airflow",
                "airflow_run_id": run_id,
            },
        )

    ingested = ingest_data()
    trained = train_model(ingested)
    metrics = evaluate_model(trained, ingested)
    gate = quality_gate(metrics)
    promoted = promote_model(trained, metrics, ingested)
    gate >> promoted


triage_training()
