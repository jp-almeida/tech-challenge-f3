"""Configuração central: rótulos, caminhos e variáveis de ambiente."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Ordem fixa: o índice da lista é o label_id usado em todo o projeto.
LABELS: list[str] = ["normal", "atencao", "urgente"]

# condition_label do dataset bruto (1-5) -> label_id (0-2).
# O mapeamento por especialidade é um proxy de urgência; ver README, seção 4.
LABEL_MAPPING: dict[int, int] = {1: 1, 2: 1, 3: 2, 4: 2, 5: 0}

RAW_FILES = {"train": "medical_tc_train.csv", "test": "medical_tc_test.csv"}

MAX_TEXT_CHARS = 20_000
MIN_TRAIN_ROWS = 2_000

DEFAULT_PARAMS: dict[str, object] = {
    "max_features": 20_000,
    "ngram_range": (1, 2),
    "min_df": 2,
    "C": 1.0,
    # False: com True a concordância de rótulos entre sklearn e ONNX cai para 98,75%.
    "sublinear_tf": False,
    "max_iter": 1000,
    "random_state": 42,
}


def _env_path(name: str, default: Path) -> Path:
    value = os.getenv(name)
    return Path(value).expanduser().resolve() if value else default


@dataclass(frozen=True)
class Settings:
    """Lida a partir das variáveis de ambiente, com defaults relativos à raiz do repo."""

    model_dir: Path = field(default_factory=lambda: _env_path("TRIAGE_MODEL_DIR", ROOT / "models"))
    data_dir: Path = field(default_factory=lambda: _env_path("TRIAGE_DATA_DIR", ROOT / "data"))
    reports_dir: Path = field(
        default_factory=lambda: _env_path("TRIAGE_REPORTS_DIR", ROOT / "reports")
    )
    backend: str = field(default_factory=lambda: os.getenv("TRIAGE_MODEL_BACKEND", "sklearn"))
    ort_threads: int = field(default_factory=lambda: int(os.getenv("TRIAGE_ORT_THREADS", "1")))
    quality_gate_f1: float = field(
        default_factory=lambda: float(os.getenv("TRIAGE_QUALITY_GATE_F1", "0.58"))
    )
    log_level: str = field(default_factory=lambda: os.getenv("TRIAGE_LOG_LEVEL", "INFO"))

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"
