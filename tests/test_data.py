from __future__ import annotations

import pandas as pd
import pytest

from triage.config import LABEL_MAPPING, MAX_TEXT_CHARS
from triage.data import SchemaError, load_raw, map_labels, normalize_text, validate_schema


def _raw(labels: list[int], text: str = "some medical abstract text") -> pd.DataFrame:
    return pd.DataFrame({"condition_label": labels, "medical_abstract": [text] * len(labels)})


def test_label_mapping_covers_all_raw_labels_and_yields_only_three_classes():
    assert set(LABEL_MAPPING) == {1, 2, 3, 4, 5}
    assert set(LABEL_MAPPING.values()) == {0, 1, 2}


def test_map_labels_follows_contract():
    mapped = map_labels(_raw([1, 2, 3, 4, 5]))
    assert mapped["label"].tolist() == [1, 1, 2, 2, 0]
    assert list(mapped.columns) == ["text", "label"]


def test_map_labels_on_fixture_is_balanced(fixture_csv):
    counts = map_labels(load_raw(fixture_csv))["label"].value_counts().to_dict()
    assert counts == {0: 30, 1: 60, 2: 60}


def test_normalize_text_strips_and_collapses_whitespace():
    assert normalize_text("  acute \n\n stroke\t with   hemiparesis  ") == (
        "acute stroke with hemiparesis"
    )


def test_normalize_text_keeps_case():
    assert normalize_text("Acute MI") == "Acute MI"


def test_normalize_text_truncates():
    assert len(normalize_text("a" * (MAX_TEXT_CHARS + 500))) == MAX_TEXT_CHARS


def test_validate_schema_rejects_missing_column():
    with pytest.raises(SchemaError, match="colunas ausentes"):
        validate_schema(pd.DataFrame({"condition_label": [1]}))


def test_validate_schema_rejects_nulls():
    df = _raw([1, 2])
    df.loc[0, "medical_abstract"] = None
    with pytest.raises(SchemaError, match="nulos"):
        validate_schema(df)


def test_validate_schema_rejects_unknown_labels():
    with pytest.raises(SchemaError, match="fora de"):
        validate_schema(_raw([1, 9]))


def test_validate_schema_enforces_min_rows():
    with pytest.raises(SchemaError, match="mínimo"):
        validate_schema(_raw([1, 2, 3]), min_rows=2000)


def test_load_raw_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_raw(tmp_path / "nope.csv")
