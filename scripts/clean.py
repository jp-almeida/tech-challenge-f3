"""Remove caches e artefatos gerados. Chamado por `make clean`.

Em Python, e não com `rm -rf`, para funcionar igual no Windows e no Linux/macOS.
"""

from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIRECTORIES = [".pytest_cache", ".ruff_cache", "htmlcov"]
FILES = [".coverage", "reports/junit.xml", "reports/coverage.xml"]
GLOBS = ["data/processed/*.csv", "models/candidates/*"]


def main() -> None:
    removed = 0
    for name in DIRECTORIES:
        target = ROOT / name
        if target.is_dir():
            shutil.rmtree(target, ignore_errors=True)
            removed += 1
    for name in FILES:
        target = ROOT / name
        if target.is_file():
            target.unlink()
            removed += 1
    for pattern in GLOBS:
        for target in ROOT.glob(pattern):
            if target.is_dir():
                shutil.rmtree(target, ignore_errors=True)
            else:
                target.unlink()
            removed += 1
    print(f"removidos: {removed} itens")


if __name__ == "__main__":
    main()
