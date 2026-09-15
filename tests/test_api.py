from __future__ import annotations

from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

from triage.api.main import create_app
from triage.api.predictor import load_predictor
from triage.config import LABELS, Settings

VALID_TEXT = "Acute myocardial infarction with ST elevation and chest pain radiating to left arm."


@pytest.fixture
def settings(trained_model_dir) -> Settings:
    return replace(Settings(), model_dir=trained_model_dir, backend="sklearn")


@pytest.fixture
def client(settings):
    # `with` dispara o lifespan (carga do modelo + warm-up).
    with TestClient(create_app(settings)) as test_client:
        yield test_client


@pytest.fixture
def client_without_model(settings, tmp_path):
    with TestClient(create_app(replace(settings, model_dir=tmp_path / "missing"))) as c:
        yield c


def test_health_ok(client):
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["backend"] == "sklearn"
    assert body["model_version"]


def test_predict_returns_contract(client):
    response = client.post("/predict", json={"text": VALID_TEXT})

    assert response.status_code == 200
    body = response.json()
    assert body["label"] in LABELS
    assert body["label_id"] == LABELS.index(body["label"])
    assert list(body["probabilities"]) == LABELS
    assert abs(sum(body["probabilities"].values()) - 1.0) < 1e-4
    assert body["label"] == max(body["probabilities"], key=body["probabilities"].get)
    assert body["backend"] == "sklearn"
    assert body["inference_ms"] >= 0


def test_predict_normalizes_whitespace(client):
    plain = client.post("/predict", json={"text": VALID_TEXT}).json()
    noisy = client.post("/predict", json={"text": f"  {VALID_TEXT.replace(' ', '   ')}\n"}).json()

    assert noisy["probabilities"] == pytest.approx(plain["probabilities"])


@pytest.mark.parametrize(
    "payload",
    [
        {"text": ""},
        {"text": "short"},
        {"text": "        padded          "},  # < 10 chars após strip
        {"text": "x" * 20_001},
        {"text": 123},
        {},
    ],
    ids=["empty", "short", "whitespace", "too-long", "not-string", "missing-field"],
)
def test_predict_validation_errors(client, payload):
    assert client.post("/predict", json=payload).status_code == 422


def test_predict_invalid_json(client):
    response = client.post(
        "/predict", content="{not json", headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 422


def test_model_info(client, trained_model_dir):
    body = client.get("/model/info").json()

    assert body["labels"] == LABELS
    assert "f1_macro" in body["metrics"]


def test_without_model_health_and_predict_are_503(client_without_model):
    health = client_without_model.get("/health")
    assert health.status_code == 503
    assert health.json()["model_loaded"] is False

    assert client_without_model.post("/predict", json={"text": VALID_TEXT}).status_code == 503
    assert client_without_model.get("/model/info").status_code == 503


def test_load_predictor_rejects_unknown_backend(trained_model_dir):
    with pytest.raises(ValueError, match="backend desconhecido"):
        load_predictor("tensorflow", trained_model_dir)
