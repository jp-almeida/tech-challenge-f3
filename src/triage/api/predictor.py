"""Backends de inferência atrás de uma interface comum.

A Etapa 7 adiciona OnnxPredictor e o registra em load_predictor.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol

import joblib
import numpy as np

from triage.train import METADATA_FILENAME, MODEL_FILENAME


class Predictor(Protocol):
    backend: str
    model_version: str
    metadata: dict[str, Any]

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        """Probabilidades shape (n, 3), colunas na ordem de LABELS."""
        ...


def read_metadata(model_dir: Path) -> dict[str, Any]:
    path = model_dir / METADATA_FILENAME
    if not path.is_file():
        raise FileNotFoundError(f"metadata não encontrado: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


class SklearnPredictor:
    backend = "sklearn"

    def __init__(self, model_dir: Path) -> None:
        model_path = model_dir / MODEL_FILENAME
        if not model_path.is_file():
            raise FileNotFoundError(f"modelo não encontrado: {model_path}")
        self.metadata = read_metadata(model_dir)
        self.model_version: str = self.metadata["model_version"]
        self._pipeline = joblib.load(model_path)

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        return self._pipeline.predict_proba(texts)


_BACKENDS = {"sklearn": SklearnPredictor}


def load_predictor(backend: str, model_dir: str | Path) -> Predictor:
    try:
        factory = _BACKENDS[backend]
    except KeyError:
        raise ValueError(
            f"backend desconhecido: {backend!r}; suportados: {sorted(_BACKENDS)}"
        ) from None
    return factory(Path(model_dir))
