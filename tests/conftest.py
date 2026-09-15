from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from triage.config import RAW_FILES

FIXTURE_CSV = Path(__file__).parent / "fixtures" / "sample_dataset.csv"
SMALL_PARAMS = {"min_df": 1, "max_features": 500, "max_iter": 200}


@pytest.fixture(scope="session")
def fixture_csv() -> Path:
    return FIXTURE_CSV


@pytest.fixture
def small_params() -> dict:
    return dict(SMALL_PARAMS)


@pytest.fixture
def project_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    """Árvore data/models/reports isolada, com a fixture no lugar dos dois CSVs brutos."""
    raw = tmp_path / "data" / "raw"
    raw.mkdir(parents=True)
    for filename in RAW_FILES.values():
        shutil.copyfile(FIXTURE_CSV, raw / filename)
    # A fixture tem 150 linhas; o mínimo de produção (2.000) é testado em test_data.
    monkeypatch.setattr("triage.data.MIN_TRAIN_ROWS", 100)
    return {
        "data_dir": tmp_path / "data",
        "model_dir": tmp_path / "models",
        "reports_dir": tmp_path / "reports",
    }


@pytest.fixture(scope="session")
def trained_model_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Modelo pequeno treinado na fixture, com model.joblib + metadata.json."""
    from triage.data import load_raw, map_labels
    from triage.train import build_pipeline, evaluate, fit, save_artifacts

    df = map_labels(load_raw(FIXTURE_CSV))
    pipeline = fit(build_pipeline(SMALL_PARAMS), df)
    out = tmp_path_factory.mktemp("model")
    save_artifacts(pipeline, evaluate(pipeline, df), out)
    return out
