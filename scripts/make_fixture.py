"""Gera tests/fixtures/sample_dataset.csv: 30 linhas por condition_label, seed 42.

Mantém o formato do CSV bruto, para que os testes exercitem load_raw/map_labels de verdade.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "raw" / "medical_tc_train.csv"
TARGET = ROOT / "tests" / "fixtures" / "sample_dataset.csv"
PER_LABEL = 30
SEED = 42


def main() -> None:
    df = pd.read_csv(SOURCE)
    sample = (
        df.groupby("condition_label", group_keys=False)
        .sample(n=PER_LABEL, random_state=SEED)
        .sort_values("condition_label", kind="stable")
    )
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(TARGET, index=False)
    print(f"{TARGET.relative_to(ROOT)}: {len(sample)} linhas")


if __name__ == "__main__":
    main()
