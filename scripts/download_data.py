"""Baixa os CSVs brutos do Medical Abstracts TC Corpus para data/raw/.

Idempotente: só baixa o que ainda não existe. Imprime tamanho, linhas e SHA-256.
"""

from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
BASE_URL = "https://raw.githubusercontent.com/sebischair/Medical-Abstracts-TC-Corpus/main"
FILES = ("medical_tc_train.csv", "medical_tc_test.csv")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(name: str) -> Path:
    target = RAW_DIR / name
    if target.exists():
        print(f"[skip] {name} já existe")
        return target
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    url = f"{BASE_URL}/{name}"
    print(f"[get ] {url}")
    tmp = target.with_suffix(".tmp")
    urllib.request.urlretrieve(url, tmp)  # noqa: S310 - URL fixa e confiável
    tmp.replace(target)
    return target


def main() -> int:
    for name in FILES:
        path = download(name)
        size_mb = path.stat().st_size / 1e6
        with path.open(encoding="utf-8") as handle:
            lines = sum(1 for _ in handle)
        print(f"       {path.relative_to(ROOT)}  {size_mb:.1f} MB  {lines} linhas")
        print(f"       sha256={sha256(path)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
