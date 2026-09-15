from __future__ import annotations

from dataclasses import replace

import joblib
import numpy as np
import pytest
from fastapi.testclient import TestClient

from triage.api.main import create_app
from triage.api.predictor import load_predictor
from triage.config import LABELS, Settings
from triage.data import load_raw, map_labels
from triage.export_onnx import ONNX_FILENAME, ParityError, check_parity
from triage.train import MODEL_FILENAME

VALID_TEXT = "Acute myocardial infarction with ST elevation and chest pain radiating to left arm."


@pytest.fixture(scope="module")
def texts(fixture_csv) -> list[str]:
    return map_labels(load_raw(fixture_csv))["text"].astype(str).tolist()


def test_onnx_matches_sklearn_on_fixture(onnx_model_dir, texts):
    pipeline = joblib.load(onnx_model_dir / MODEL_FILENAME)

    report = check_parity(pipeline, onnx_model_dir / ONNX_FILENAME, texts)

    assert report["label_agreement"] >= 0.99
    assert report["n"] == len(texts)


def test_check_parity_raises_when_threshold_not_met(onnx_model_dir, texts, monkeypatch):
    pipeline = joblib.load(onnx_model_dir / MODEL_FILENAME)
    monkeypatch.setattr("triage.export_onnx.MIN_LABEL_AGREEMENT", 1.01)

    with pytest.raises(ParityError, match="label_agreement"):
        check_parity(pipeline, onnx_model_dir / ONNX_FILENAME, texts)


def test_onnx_predictor_agrees_with_sklearn(onnx_model_dir, texts):
    sklearn_predictor = load_predictor("sklearn", onnx_model_dir)
    onnx_predictor = load_predictor("onnx", onnx_model_dir)

    sklearn_proba = sklearn_predictor.predict_proba(texts[:20])
    onnx_proba = onnx_predictor.predict_proba(texts[:20])

    assert onnx_predictor.backend == "onnx"
    assert onnx_proba.shape == (20, len(LABELS))
    np.testing.assert_allclose(onnx_proba, sklearn_proba, atol=1e-3)


def test_onnx_predictor_requires_onnx_file(trained_model_dir):
    with pytest.raises(FileNotFoundError, match="ONNX"):
        load_predictor("onnx", trained_model_dir)


def test_api_serves_onnx_backend(onnx_model_dir):
    settings = replace(Settings(), model_dir=onnx_model_dir, backend="onnx")

    with TestClient(create_app(settings)) as client:
        health = client.get("/health").json()
        body = client.post("/predict", json={"text": VALID_TEXT}).json()

    assert health["backend"] == "onnx"
    assert body["backend"] == "onnx"
    assert body["label"] in LABELS
    assert abs(sum(body["probabilities"].values()) - 1.0) < 1e-4


def test_api_predictions_match_across_backends(onnx_model_dir):
    base = replace(Settings(), model_dir=onnx_model_dir)
    payload = {"text": VALID_TEXT}

    with TestClient(create_app(replace(base, backend="sklearn"))) as client:
        sklearn_body = client.post("/predict", json=payload).json()
    with TestClient(create_app(replace(base, backend="onnx"))) as client:
        onnx_body = client.post("/predict", json=payload).json()

    assert onnx_body["label"] == sklearn_body["label"]
    for label in LABELS:
        assert onnx_body["probabilities"][label] == pytest.approx(
            sklearn_body["probabilities"][label], abs=1e-3
        )
