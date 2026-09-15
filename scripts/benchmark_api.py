"""Benchmark de latência ponta a ponta (HTTP) do POST /predict.

Reutilizado sem mudanças na Etapa 7 para comparar backends:
    python scripts/benchmark_api.py --label sklearn
    python scripts/benchmark_api.py --label onnx
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TEST_CSV = ROOT / "data" / "raw" / "medical_tc_test.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--n", type=int, default=1000, help="requisições medidas")
    parser.add_argument("--warmup", type=int, default=50, help="requisições descartadas")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--label", required=True, help="identificador do run, ex.: sklearn")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, help="padrão: reports/latency/api_<label>.json")
    parser.add_argument(
        "--docker",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="a API medida roda em container (registrado no relatório)",
    )
    return parser.parse_args()


def sample_texts(n: int, seed: int) -> list[str]:
    texts = pd.read_csv(TEST_CSV)["medical_abstract"].astype(str).tolist()
    rng = random.Random(seed)
    return [rng.choice(texts) for _ in range(n)]


def percentile(sorted_values: list[float], q: float) -> float:
    """Percentil por interpolação linear (equivalente ao numpy 'linear')."""
    pos = (len(sorted_values) - 1) * q
    low = int(pos)
    high = min(low + 1, len(sorted_values) - 1)
    return sorted_values[low] + (sorted_values[high] - sorted_values[low]) * (pos - low)


def cpu_model() -> str:
    try:
        if sys.platform == "darwin":
            cmd = ["sysctl", "-n", "machdep.cpu.brand_string"]
            return subprocess.check_output(cmd, text=True).strip()
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return platform.processor() or "unknown"


def main() -> int:
    args = parse_args()
    out = args.out or ROOT / "reports" / "latency" / f"api_{args.label}.json"
    texts = sample_texts(args.warmup + args.n, args.seed)
    limits = httpx.Limits(max_keepalive_connections=args.concurrency)

    with httpx.Client(base_url=args.url, timeout=10.0, limits=limits) as client:
        health = client.get("/health").raise_for_status().json()

        def call(text: str) -> tuple[float, bool]:
            start = time.perf_counter_ns()
            try:
                ok = client.post("/predict", json={"text": text}).status_code == 200
            except httpx.HTTPError:
                ok = False
            return (time.perf_counter_ns() - start) / 1e6, ok

        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            list(pool.map(call, texts[: args.warmup]))
            wall_start = time.perf_counter()
            results = list(pool.map(call, texts[args.warmup :]))
            wall_s = time.perf_counter() - wall_start

    latencies = sorted(ms for ms, ok in results if ok)
    errors = sum(1 for _, ok in results if not ok)
    if not latencies:
        print("todas as requisições falharam", file=sys.stderr)
        return 1

    report = {
        "label": args.label,
        "n": len(results),
        "errors": errors,
        "concurrency": args.concurrency,
        "warmup": args.warmup,
        "mean_ms": round(statistics.fmean(latencies), 4),
        "p50_ms": round(percentile(latencies, 0.50), 4),
        "p95_ms": round(percentile(latencies, 0.95), 4),
        "p99_ms": round(percentile(latencies, 0.99), 4),
        "min_ms": round(latencies[0], 4),
        "max_ms": round(latencies[-1], 4),
        "throughput_rps": round(len(results) / wall_s, 2),
        "environment": {
            "platform": platform.platform(),
            "cpu": cpu_model(),
            "docker": args.docker,
            "url": args.url,
            "measured_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "model_version": health.get("model_version"),
            "backend": health.get("backend"),
            "seed": args.seed,
        },
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "environment"}, indent=2))
    print(f"-> {out.relative_to(ROOT) if out.is_relative_to(ROOT) else out}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
