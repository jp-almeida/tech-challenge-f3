"""Construção, treino, avaliação e persistência do pipeline TF-IDF + LogisticRegression."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import Pipeline

from triage.config import DEFAULT_PARAMS, LABELS

MODEL_FILENAME = "model.joblib"
METADATA_FILENAME = "metadata.json"


def build_pipeline(params: dict[str, Any] | None = None) -> Pipeline:
    """Vetorizador sem stop_words/strip_accents/tokenizer custom — exigência da conversão ONNX."""
    p = {**DEFAULT_PARAMS, **(params or {})}
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=tuple(p["ngram_range"]),
                    min_df=p["min_df"],
                    max_features=p["max_features"],
                    sublinear_tf=p["sublinear_tf"],
                    dtype=np.float32,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    C=p["C"],
                    max_iter=p["max_iter"],
                    random_state=p["random_state"],
                ),
            ),
        ]
    )


def fit(pipeline: Pipeline, df: pd.DataFrame) -> Pipeline:
    pipeline.fit(df["text"].tolist(), df["label"].to_numpy())
    return pipeline


def evaluate(pipeline: Pipeline, df: pd.DataFrame) -> dict[str, Any]:
    y_true = df["label"].to_numpy()
    y_pred = pipeline.predict(df["text"].tolist())
    report = classification_report(
        y_true, y_pred, labels=[0, 1, 2], target_names=LABELS, output_dict=True, zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_per_class": {name: float(report[name]["f1-score"]) for name in LABELS},
        "classification_report": report,
        "n_eval": int(len(df)),
    }


def build_metadata(
    pipeline: Pipeline, metrics: dict[str, Any], extra: dict[str, Any] | None = None
) -> dict[str, Any]:
    from triage.config import LABEL_MAPPING

    now = datetime.now(UTC)
    tfidf: TfidfVectorizer = pipeline.named_steps["tfidf"]
    clf: LogisticRegression = pipeline.named_steps["clf"]
    metadata: dict[str, Any] = {
        "model_version": now.strftime("%Y%m%dT%H%M%SZ"),
        "trained_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "dataset": "medical-abstracts-tc-corpus",
        "label_mapping": {str(k): LABELS[v] for k, v in LABEL_MAPPING.items()},
        "labels": LABELS,
        "params": {
            "max_features": tfidf.max_features,
            "ngram_range": list(tfidf.ngram_range),
            "min_df": tfidf.min_df,
            "sublinear_tf": tfidf.sublinear_tf,
            "C": clf.C,
        },
        "metrics": {
            "accuracy": metrics["accuracy"],
            "f1_macro": metrics["f1_macro"],
            "f1_per_class": metrics["f1_per_class"],
        },
        "versions": _versions(),
        "onnx": {"available": False, "parity_label_agreement": None, "max_abs_proba_diff": None},
    }
    metadata.update(extra or {})
    return metadata


def _versions() -> dict[str, str]:
    import platform
    from importlib.metadata import PackageNotFoundError, version

    versions = {"python": platform.python_version()}
    for package in ("scikit-learn", "skl2onnx", "onnxruntime"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def save_artifacts(
    pipeline: Pipeline,
    metrics: dict[str, Any],
    out_dir: str | Path,
    extra: dict[str, Any] | None = None,
) -> dict[str, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model_path = out / MODEL_FILENAME
    metadata_path = out / METADATA_FILENAME
    joblib.dump(pipeline, model_path)
    metadata_path.write_text(
        json.dumps(build_metadata(pipeline, metrics, extra), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return {"model_path": model_path, "metadata_path": metadata_path}
