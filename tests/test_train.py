from __future__ import annotations

import json

import joblib
import numpy as np

from triage.config import LABELS
from triage.data import load_raw, map_labels
from triage.train import build_pipeline, evaluate, fit, save_artifacts

METADATA_KEYS = {
    "model_version",
    "trained_at",
    "dataset",
    "label_mapping",
    "labels",
    "params",
    "metrics",
    "versions",
    "onnx",
}


def test_pipeline_predicts_probabilities_in_label_order(fixture_csv, small_params):
    df = map_labels(load_raw(fixture_csv))
    pipeline = fit(build_pipeline(small_params), df)

    proba = pipeline.predict_proba(df["text"].head(7).tolist())

    assert proba.shape == (7, len(LABELS))
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)
    assert pipeline.named_steps["clf"].classes_.tolist() == [0, 1, 2]


def test_evaluate_returns_contract_metrics(fixture_csv, small_params):
    df = map_labels(load_raw(fixture_csv))
    metrics = evaluate(fit(build_pipeline(small_params), df), df)

    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["f1_macro"] <= 1.0
    assert set(metrics["f1_per_class"]) == set(LABELS)


def test_save_artifacts_writes_model_and_metadata(trained_model_dir):
    metadata = json.loads((trained_model_dir / "metadata.json").read_text())

    assert METADATA_KEYS <= set(metadata)
    assert metadata["labels"] == LABELS
    assert metadata["onnx"]["available"] is False
    assert set(metadata["metrics"]["f1_per_class"]) == set(LABELS)
    assert hasattr(joblib.load(trained_model_dir / "model.joblib"), "predict_proba")


def test_save_artifacts_merges_extra(tmp_path, fixture_csv, small_params):
    df = map_labels(load_raw(fixture_csv))
    pipeline = fit(build_pipeline(small_params), df)

    paths = save_artifacts(pipeline, evaluate(pipeline, df), tmp_path, extra={"n_train": 150})

    assert json.loads(paths["metadata_path"].read_text())["n_train"] == 150
