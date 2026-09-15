"""Integridade da DAG. Pulado quando o Airflow não está instalado (ex.: CI e .venv local).

Para rodar dentro da imagem do Airflow: `make test-dag`.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("airflow")

from airflow.dag_processing.dagbag import DagBag  # noqa: E402

pytestmark = pytest.mark.airflow

DAG_FOLDER = Path(__file__).resolve().parents[1] / "airflow" / "dags"
EXPECTED_CHAIN = [
    "ingest_data",
    "train_model",
    "evaluate_model",
    "quality_gate",
    "export_onnx",
    "promote_model",
]


@pytest.fixture(scope="module")
def dag():
    # Airflow 3.3 removeu `include_examples`; só a pasta indicada é carregada.
    dagbag = DagBag(dag_folder=str(DAG_FOLDER))
    assert dagbag.import_errors == {}
    return dagbag.dags["triage_training"]


def test_dag_has_expected_tasks(dag):
    assert sorted(dag.task_ids) == sorted(EXPECTED_CHAIN)


def test_dag_task_order(dag):
    for upstream, downstream in zip(EXPECTED_CHAIN, EXPECTED_CHAIN[1:], strict=False):
        assert downstream in dag.get_task(upstream).downstream_task_ids


def test_promote_never_runs_without_the_quality_gate(dag):
    # quality_gate é upstream indireto (quality_gate -> export_onnx -> promote_model).
    upstream = {t.task_id for t in dag.get_task("promote_model").get_flat_relatives(upstream=True)}

    assert {"quality_gate", "export_onnx", "train_model", "evaluate_model", "ingest_data"} <= (
        upstream
    )


def test_dag_is_manual_and_has_params(dag):
    assert dag.schedule is None
    assert dag.catchup is False
    assert {"max_features", "C", "min_f1"} <= set(dag.params)
    assert dag.get_task("quality_gate").retries == 0
