"""Gera tráfego contra a API para os painéis do Grafana terem o que mostrar.

Inclui requisições inválidas de propósito (422) e rota inexistente (404), para o
painel de taxa de erro não ficar zerado:

    python scripts/load_test.py --duration 120 --rps 20
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from collections import Counter
from pathlib import Path

import httpx
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TEST_CSV = ROOT / "data" / "raw" / "medical_tc_test.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--duration", type=float, default=120, help="segundos")
    parser.add_argument("--rps", type=float, default=20, help="requisições por segundo")
    parser.add_argument("--error-ratio", type=float, default=0.05, help="fração de erros (0-1)")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def send_invalid(client: httpx.Client, rng: random.Random) -> httpx.Response:
    kind = rng.choice(["empty_text", "missing_field", "unknown_route"])
    if kind == "empty_text":
        return client.post("/predict", json={"text": ""})
    if kind == "missing_field":
        return client.post("/predict", json={})
    return client.get("/rota-inexistente")


def main() -> int:
    args = parse_args()
    rng = random.Random(args.seed)
    texts = pd.read_csv(TEST_CSV)["medical_abstract"].astype(str).tolist()

    interval = 1.0 / args.rps
    deadline = time.perf_counter() + args.duration
    statuses: Counter[str] = Counter()
    latencies: list[float] = []

    print(f"enviando ~{args.rps} req/s por {args.duration:.0f}s em {args.url} (ctrl-c para parar)")
    with httpx.Client(base_url=args.url, timeout=10.0) as client:
        try:
            while time.perf_counter() < deadline:
                cycle_start = time.perf_counter()
                try:
                    if rng.random() < args.error_ratio:
                        response = send_invalid(client, rng)
                    else:
                        response = client.post("/predict", json={"text": rng.choice(texts)})
                    statuses[str(response.status_code)] += 1
                except httpx.HTTPError as exc:
                    statuses[type(exc).__name__] += 1
                latencies.append(time.perf_counter() - cycle_start)

                sleep_for = interval - (time.perf_counter() - cycle_start)
                if sleep_for > 0:
                    time.sleep(sleep_for)
        except KeyboardInterrupt:
            print("\ninterrompido")

    total = sum(statuses.values())
    print(f"\ntotal: {total} requisições")
    for status, count in sorted(statuses.items()):
        print(f"  {status}: {count} ({100 * count / total:.1f}%)")
    if latencies:
        print(f"latência média (cliente): {1000 * sum(latencies) / len(latencies):.2f} ms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
