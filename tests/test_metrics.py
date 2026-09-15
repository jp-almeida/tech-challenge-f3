"""Os contadores do prometheus_client são globais ao processo, então cada teste compara
o valor antes e depois da requisição em vez de assumir que começa em zero.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
from fastapi.testclient import TestClient
from prometheus_client import REGISTRY

from triage.api.main import create_app
from triage.config import LABELS, Settings

VALID_TEXT = "Acute myocardial infarction with ST elevation and chest pain radiating to left arm."


@pytest.fixture
def client(trained_model_dir):
    settings = replace(Settings(), model_dir=trained_model_dir, backend="sklearn")
    with TestClient(create_app(settings)) as test_client:
        yield test_client


def sample(name: str, **labels) -> float:
    return REGISTRY.get_sample_value(name, labels) or 0.0


def requests_total(handler: str, status: str, method: str = "POST") -> float:
    return sample("triage_http_requests_total", method=method, handler=handler, status=status)


def test_predict_increments_request_and_prediction_counters(client):
    before_requests = requests_total("/predict", "200")
    before_predictions = sum(
        sample("triage_predictions_total", label=label, backend="sklearn") for label in LABELS
    )

    body = client.post("/predict", json={"text": VALID_TEXT}).json()

    assert requests_total("/predict", "200") == before_requests + 1
    after_predictions = sum(
        sample("triage_predictions_total", label=label, backend="sklearn") for label in LABELS
    )
    assert after_predictions == before_predictions + 1
    assert sample("triage_predictions_total", label=body["label"], backend="sklearn") >= 1


def test_metrics_endpoint_exposes_contract_metrics(client):
    client.post("/predict", json={"text": VALID_TEXT})

    response = client.get("/metrics")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    body = response.text
    for metric in (
        "triage_http_requests_total",
        "triage_http_request_duration_seconds",
        "triage_model_inference_duration_seconds",
        "triage_predictions_total",
        "triage_model_info",
    ):
        assert metric in body
    assert 'triage_http_requests_total{handler="/predict",method="POST",status="200"}' in body


def test_inference_histogram_is_observed(client):
    before = sample("triage_model_inference_duration_seconds_count", backend="sklearn")

    client.post("/predict", json={"text": VALID_TEXT})

    assert sample("triage_model_inference_duration_seconds_count", backend="sklearn") == before + 1


def test_model_info_gauge_has_version_label(client):
    version = client.get("/health").json()["model_version"]

    assert sample("triage_model_info", model_version=version, backend="sklearn") == 1


def test_validation_error_is_counted_as_422(client):
    before = requests_total("/predict", "422")

    client.post("/predict", json={"text": "short"})

    assert requests_total("/predict", "422") == before + 1


def test_unknown_path_does_not_explode_cardinality(client):
    before = requests_total(handler="unmatched", status="404", method="GET")

    client.get("/nao-existe/123")
    client.get("/nao-existe/456")

    assert requests_total(handler="unmatched", status="404", method="GET") == before + 2
    assert sample("triage_http_requests_total", method="GET", handler="/nao-existe/123") == 0.0


def test_unhandled_exception_is_counted_as_500(client):
    @client.app.get("/boom")
    def boom():
        raise RuntimeError("falha inesperada")

    before = requests_total(handler="/boom", status="500", method="GET")

    with pytest.raises(RuntimeError):
        client.get("/boom")

    assert requests_total(handler="/boom", status="500", method="GET") == before + 1


def test_metrics_endpoint_is_not_self_counted(client):
    client.get("/metrics")

    assert requests_total(handler="/metrics", status="200", method="GET") == 0.0


def test_http_duration_histogram_records_handler_template(client):
    before = sample("triage_http_request_duration_seconds_count", method="POST", handler="/predict")

    client.post("/predict", json={"text": VALID_TEXT})

    after = sample("triage_http_request_duration_seconds_count", method="POST", handler="/predict")
    assert after == before + 1
