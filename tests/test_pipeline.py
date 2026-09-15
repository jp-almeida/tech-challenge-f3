from __future__ import annotations

import json

import pandas as pd
import pytest

from triage import pipeline
from triage.config import Settings


def _settings(dirs, **overrides) -> Settings:
    return Settings(
        model_dir=dirs["model_dir"],
        data_dir=dirs["data_dir"],
        reports_dir=dirs["reports_dir"],
        **{"quality_gate_f1": 0.0, **overrides},
    )


def test_ingest_writes_processed_csvs(project_dirs):
    result = pipeline.ingest(project_dirs["data_dir"])

    train = pd.read_csv(result["train_path"])
    assert list(train.columns) == ["text", "label"]
    assert result["n_train"] == result["n_test"] == 150
    assert result["class_counts"] == {"normal": 30, "atencao": 60, "urgente": 60}
    json.dumps(result)  # precisa ser serializável para XCom


def test_run_all_promotes_model(project_dirs, small_params):
    result = pipeline.run_all(_settings(project_dirs), small_params)

    model_dir = project_dirs["model_dir"]
    metadata = json.loads((model_dir / "metadata.json").read_text())
    assert (model_dir / "model.joblib").is_file()
    assert metadata["model_version"] == result["model_version"]
    assert metadata["n_train"] == metadata["n_test"] == 150
    assert (project_dirs["reports_dir"] / "metrics" / "classification_report.json").is_file()
    assert not list(model_dir.glob("*.tmp"))


def test_run_all_stops_at_quality_gate(project_dirs, small_params):
    with pytest.raises(pipeline.QualityGateError):
        pipeline.run_all(_settings(project_dirs, quality_gate_f1=1.01), small_params)

    assert not (project_dirs["model_dir"] / "metadata.json").exists()


def test_quality_gate_threshold():
    pipeline.quality_gate({"f1_macro": 0.60}, min_f1=0.58)
    with pytest.raises(pipeline.QualityGateError, match="abaixo do mínimo"):
        pipeline.quality_gate({"f1_macro": 0.55}, min_f1=0.58)


def test_cli_ingest_uses_env_vars(project_dirs, monkeypatch, capsys):
    monkeypatch.setenv("TRIAGE_DATA_DIR", str(project_dirs["data_dir"]))

    assert pipeline.main(["ingest"]) == 0

    assert json.loads(capsys.readouterr().out)["n_train"] == 150
    assert (project_dirs["data_dir"] / "processed" / "test.csv").is_file()
