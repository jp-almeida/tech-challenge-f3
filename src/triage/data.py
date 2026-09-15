"""Carga, validação de schema, mapeamento de rótulos e normalização de texto."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from triage.config import LABEL_MAPPING, MAX_TEXT_CHARS, MIN_TRAIN_ROWS

RAW_COLUMNS = ("condition_label", "medical_abstract")
_WHITESPACE = re.compile(r"\s+")


class SchemaError(ValueError):
    """Dataset bruto não atende ao contrato esperado."""


def normalize_text(text: str) -> str:
    """Strip, colapsa espaços em branco e trunca. Lowercase fica a cargo do vetorizador."""
    return _WHITESPACE.sub(" ", str(text)).strip()[:MAX_TEXT_CHARS]


def validate_schema(df: pd.DataFrame, *, min_rows: int = 0) -> None:
    missing = [c for c in RAW_COLUMNS if c not in df.columns]
    if missing:
        raise SchemaError(f"colunas ausentes: {missing}; esperado {list(RAW_COLUMNS)}")
    if df[list(RAW_COLUMNS)].isna().any().any():
        raise SchemaError("há valores nulos em condition_label/medical_abstract")
    unknown = sorted(set(df["condition_label"].unique()) - set(LABEL_MAPPING))
    if unknown:
        raise SchemaError(f"condition_label fora de {sorted(LABEL_MAPPING)}: {unknown}")
    if len(df) < min_rows:
        raise SchemaError(f"dataset tem {len(df)} linhas, mínimo exigido é {min_rows}")


def load_raw(path: str | Path, *, min_rows: int = 0) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV bruto não encontrado: {path}")
    df = pd.read_csv(path)
    validate_schema(df, min_rows=min_rows)
    return df


def map_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Devolve um DataFrame com apenas as colunas `text` (normalizada) e `label` (0/1/2)."""
    out = pd.DataFrame(
        {
            "text": df["medical_abstract"].map(normalize_text),
            "label": df["condition_label"].astype(int).map(LABEL_MAPPING).astype(int),
        }
    )
    return out[out["text"].str.len() > 0].reset_index(drop=True)


def load_processed(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(Path(path))
    if not {"text", "label"} <= set(df.columns):
        raise SchemaError(
            f"CSV processado precisa de colunas text,label; tem {df.columns.tolist()}"
        )
    df["text"] = df["text"].fillna("").astype(str)
    return df


def prepare(raw_path: str | Path, *, is_train: bool) -> pd.DataFrame:
    """load_raw + map_labels, aplicando o mínimo de linhas só no conjunto de treino."""
    return map_labels(load_raw(raw_path, min_rows=MIN_TRAIN_ROWS if is_train else 0))
