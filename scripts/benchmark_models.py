"""Benchmark in-process (sem HTTP) do pipeline sklearn contra o ONNX Runtime.

Ambos recebem exatamente os mesmos textos já normalizados e passam pelo mesmo warm-up.
Rode com o mínimo de coisas concorrentes na máquina (sem Airflow, sem load test):

    OMP_NUM_THREADS=1 python scripts/benchmark_models.py
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
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

TEST_CSV = ROOT / "data" / "raw" / "medical_tc_test.csv"
BATCH_ITERATIONS = {1: 2000, 32: 200}
WARMUP = 100


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "reports" / "latency")
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def percentile(sorted_values: list[float], q: float) -> float:
    pos = (len(sorted_values) - 1) * q
    low = int(pos)
    high = min(low + 1, len(sorted_values) - 1)
    return sorted_values[low] + (sorted_values[high] - sorted_values[low]) * (pos - low)


def summarize(samples_ms: list[float], batch: int) -> dict[str, float]:
    ordered = sorted(samples_ms)
    mean = statistics.fmean(ordered)
    return {
        "mean_ms": round(mean, 4),
        "p50_ms": round(percentile(ordered, 0.50), 4),
        "p95_ms": round(percentile(ordered, 0.95), 4),
        "p99_ms": round(percentile(ordered, 0.99), 4),
        "min_ms": round(ordered[0], 4),
        "max_ms": round(ordered[-1], 4),
        "throughput_rps": round(1000 * batch / mean, 1),
        "iterations": len(ordered),
    }


def measure(predict, batches: list[list[str]], iterations: int) -> list[float]:
    for batch in batches[:WARMUP]:
        predict(batch)
    samples = []
    for i in range(iterations):
        batch = batches[i % len(batches)]
        start = time.perf_counter_ns()
        predict(batch)
        samples.append((time.perf_counter_ns() - start) / 1e6)
    return samples


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


def write_markdown(report: dict, path: Path) -> None:
    env = report["environment"]
    lines = [
        "# Comparação de latência — sklearn vs. ONNX Runtime",
        "",
        f"Medido em {env['measured_at']} · modelo `{env['model_version']}`.",
        "",
        "| Cenário | Backend | Batch | p50 (ms) | p95 (ms) | p99 (ms) | Throughput (req/s) "
        "| Speedup p50 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for batch in sorted(BATCH_ITERATIONS):
        base = report["in_process"]["sklearn"][str(batch)]
        for backend in ("sklearn", "onnx"):
            row = report["in_process"][backend][str(batch)]
            speedup = base["p50_ms"] / row["p50_ms"]
            lines.append(
                f"| In-process | {backend} | {batch} | {row['p50_ms']:.4f} | {row['p95_ms']:.4f} "
                f"| {row['p99_ms']:.4f} | {row['throughput_rps']:.0f} | {speedup:.2f}× |"
            )

    http = report.get("http", {})
    if "sklearn" in http:
        base_http = http["sklearn"]
        for backend in ("sklearn", "onnx"):
            row = http.get(backend)
            if not row:
                continue
            speedup = base_http["p50_ms"] / row["p50_ms"]
            lines.append(
                f"| HTTP (Docker) | {backend} | 1 | {row['p50_ms']:.4f} | {row['p95_ms']:.4f} "
                f"| {row['p99_ms']:.4f} | {row['throughput_rps']:.0f} | {speedup:.2f}× |"
            )
    else:
        lines.append(
            "| HTTP (Docker) | — | — | — | — | — | — | *rode `make bench-api` nos dois backends* |"
        )

    parity = report["parity"]
    sizes = report["artifact_bytes"]
    lines += [
        "",
        "## Paridade",
        "",
        f"- Concordância de rótulos: **{100 * parity['label_agreement']:.2f}%** "
        f"em {parity['n']} textos do conjunto de teste.",
        f"- Diferença de probabilidade: mediana {parity['median_abs_proba_diff']:.2e}, "
        f"p99,9 {parity['p999_abs_proba_diff']:.2e}, máxima {parity['max_abs_proba_diff']:.2e} "
        f"({parity['n_above_1e_3']} textos acima de 1e-3).",
        "",
        "## Artefatos",
        "",
        f"- `model.joblib`: {sizes['model.joblib'] / 1e6:.2f} MB",
        f"- `model.onnx`: {sizes['model.onnx'] / 1e6:.2f} MB",
        "",
        "## Ambiente",
        "",
        f"- CPU: {env['cpu']}",
        f"- Plataforma: {env['platform']}",
        f"- Versões: {', '.join(f'{k} {v}' for k, v in env['versions'].items())}",
        f"- Threads: intra_op=1, OMP_NUM_THREADS={env['omp_num_threads']}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_plot(report: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    batches = sorted(BATCH_ITERATIONS)
    fig, axes = plt.subplots(1, len(batches), figsize=(10, 4))
    for ax, batch in zip(axes, batches, strict=True):
        values = [report["in_process"][b][str(batch)] for b in ("sklearn", "onnx")]
        x = range(2)
        ax.bar([i - 0.2 for i in x], [v["p50_ms"] for v in values], 0.4, label="p50")
        ax.bar([i + 0.2 for i in x], [v["p95_ms"] for v in values], 0.4, label="p95")
        ax.set_xticks(list(x))
        ax.set_xticklabels(["sklearn", "onnx"])
        ax.set_ylabel("ms por chamada")
        ax.set_title(f"batch {batch}")
        ax.legend()
    fig.suptitle("Latência in-process: sklearn vs. ONNX Runtime")
    fig.tight_layout()
    fig.savefig(path, dpi=130)


def main() -> int:
    import os

    import joblib
    import pandas as pd

    from triage.data import normalize_text
    from triage.export_onnx import ONNX_FILENAME, check_parity, make_session, run_onnx
    from triage.train import MODEL_FILENAME

    args = parse_args()
    model_dir: Path = args.model_dir
    pipeline = joblib.load(model_dir / MODEL_FILENAME)
    session = make_session(model_dir / ONNX_FILENAME, threads=1)

    raw = pd.read_csv(TEST_CSV)["medical_abstract"].astype(str).tolist()
    texts = [normalize_text(t) for t in raw]
    rng = random.Random(args.seed)

    report: dict = {"in_process": {"sklearn": {}, "onnx": {}}}
    for batch, iterations in BATCH_ITERATIONS.items():
        batches = [[rng.choice(texts) for _ in range(batch)] for _ in range(50)]
        report["in_process"]["sklearn"][str(batch)] = summarize(
            measure(pipeline.predict_proba, batches, iterations), batch
        )
        report["in_process"]["onnx"][str(batch)] = summarize(
            measure(lambda b: run_onnx(session, b), batches, iterations), batch
        )
        speedup = (
            report["in_process"]["sklearn"][str(batch)]["p50_ms"]
            / report["in_process"]["onnx"][str(batch)]["p50_ms"]
        )
        print(f"batch {batch:>2}: speedup p50 = {speedup:.2f}×")

    report["parity"] = check_parity(pipeline, model_dir / ONNX_FILENAME, texts)
    report["artifact_bytes"] = {
        MODEL_FILENAME: (model_dir / MODEL_FILENAME).stat().st_size,
        ONNX_FILENAME: (model_dir / ONNX_FILENAME).stat().st_size,
    }
    report["environment"] = {
        "cpu": cpu_model(),
        "platform": platform.platform(),
        "measured_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model_version": json.loads((model_dir / "metadata.json").read_text())["model_version"],
        "omp_num_threads": os.getenv("OMP_NUM_THREADS", "não definido"),
        "seed": args.seed,
        "versions": _versions(),
    }

    # Incorpora os benchmarks HTTP da Etapa 2/7, se já existirem.
    http = {}
    for backend in ("sklearn", "onnx"):
        path = args.out_dir / f"api_{backend}.json"
        if path.is_file():
            http[backend] = json.loads(path.read_text())
    report["http"] = http

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "models_comparison.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    write_markdown(report, args.out_dir / "comparison.md")
    if not args.no_plot:
        write_plot(report, args.out_dir / "latency_comparison.png")
    print(f"-> {args.out_dir.relative_to(ROOT)}/{{models_comparison.json,comparison.md}}")
    return 0


def _versions() -> dict[str, str]:
    from importlib.metadata import PackageNotFoundError, version

    out = {"python": platform.python_version()}
    for package in ("scikit-learn", "skl2onnx", "onnxruntime", "numpy"):
        try:
            out[package] = version(package)
        except PackageNotFoundError:
            out[package] = "not-installed"
    return out


if __name__ == "__main__":
    sys.exit(main())
