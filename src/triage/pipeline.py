"""Passos orquestráveis do treino, usados pela CLI (`make train`) e pela DAG do Airflow.

Contrato (§3.4 do guia): I/O apenas por caminhos; retornos pequenos e serializáveis em
JSON, para trafegarem por XCom sem carregar DataFrames.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

from triage.config import DEFAULT_PARAMS, LABELS, RAW_FILES, Settings

log = logging.getLogger("triage.pipeline")


class QualityGateError(RuntimeError):
    """F1 macro do candidato abaixo do limiar mínimo."""


def ingest(data_dir: str | Path) -> dict[str, Any]:
    """Garante os CSVs brutos, valida, mapeia rótulos e grava data/processed/{train,test}.csv."""
    from triage.data import prepare

    data_dir = Path(data_dir)
    raw_dir, processed_dir = data_dir / "raw", data_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    missing = [n for n in RAW_FILES.values() if not (raw_dir / n).exists()]
    if missing:
        log.info("CSVs ausentes (%s), baixando", ", ".join(missing))
        _download_raw(raw_dir)

    out: dict[str, Any] = {}
    for split, filename in RAW_FILES.items():
        df = prepare(raw_dir / filename, is_train=split == "train")
        path = processed_dir / f"{split}.csv"
        df.to_csv(path, index=False)
        out[f"{split}_path"] = str(path)
        out[f"n_{split}"] = int(len(df))
        if split == "train":
            counts = df["label"].value_counts().sort_index()
            out["class_counts"] = {LABELS[i]: int(counts.get(i, 0)) for i in range(len(LABELS))}
    log.info("ingest concluído: %s", {k: v for k, v in out.items() if not k.endswith("_path")})
    return out


def _download_raw(raw_dir: Path) -> None:
    import urllib.request

    base = "https://raw.githubusercontent.com/sebischair/Medical-Abstracts-TC-Corpus/main"
    raw_dir.mkdir(parents=True, exist_ok=True)
    for name in RAW_FILES.values():
        target = raw_dir / name
        if not target.exists():
            urllib.request.urlretrieve(f"{base}/{name}", target)  # noqa: S310


def train(
    train_path: str | Path, candidate_dir: str | Path, params: dict[str, Any] | None = None
) -> dict[str, Any]:
    import joblib

    from triage.data import load_processed
    from triage.train import MODEL_FILENAME, build_pipeline, fit

    candidate_dir = Path(candidate_dir)
    candidate_dir.mkdir(parents=True, exist_ok=True)
    pipeline = fit(build_pipeline(params), load_processed(train_path))
    model_path = candidate_dir / MODEL_FILENAME
    joblib.dump(pipeline, model_path)
    log.info("modelo treinado em %s", model_path)
    return {"model_path": str(model_path)}


def evaluate(
    model_path: str | Path, test_path: str | Path, reports_dir: str | Path
) -> dict[str, Any]:
    import joblib

    from triage.data import load_processed
    from triage.train import evaluate as evaluate_pipeline

    metrics = evaluate_pipeline(joblib.load(model_path), load_processed(test_path))
    metrics_dir = Path(reports_dir) / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    (metrics_dir / "classification_report.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    log.info("avaliação: accuracy=%.4f f1_macro=%.4f", metrics["accuracy"], metrics["f1_macro"])
    # Retorno enxuto para XCom: o relatório completo fica no arquivo.
    return {
        "accuracy": metrics["accuracy"],
        "f1_macro": metrics["f1_macro"],
        "f1_per_class": metrics["f1_per_class"],
    }


def quality_gate(metrics: dict[str, Any], min_f1: float) -> None:
    f1 = float(metrics["f1_macro"])
    if f1 < min_f1:
        raise QualityGateError(f"f1_macro {f1:.4f} abaixo do mínimo {min_f1:.4f}")
    log.info("quality gate aprovado: f1_macro=%.4f >= %.4f", f1, min_f1)


def export_onnx(
    model_path: str | Path, test_path: str | Path, candidate_dir: str | Path
) -> dict[str, Any]:
    """Converte o candidato para ONNX e valida a paridade contra o sklearn."""
    import joblib

    from triage.data import load_processed
    from triage.export_onnx import ONNX_FILENAME, check_parity, convert_to_onnx

    candidate_dir = Path(candidate_dir)
    candidate_dir.mkdir(parents=True, exist_ok=True)
    pipeline = joblib.load(model_path)
    onnx_path = candidate_dir / ONNX_FILENAME
    onnx_path.write_bytes(convert_to_onnx(pipeline))

    texts = load_processed(test_path)["text"].astype(str).tolist()
    parity = check_parity(pipeline, onnx_path, texts)
    log.info(
        "onnx exportado: %s (%.2f MB) paridade=%.4f mediana_dif=%.2e",
        onnx_path,
        onnx_path.stat().st_size / 1e6,
        parity["label_agreement"],
        parity["median_abs_proba_diff"],
    )
    return {"onnx_path": str(onnx_path), **parity}


def promote(
    candidate_dir: str | Path,
    model_dir: str | Path,
    metrics: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Copia os artefatos do candidato para models/ de forma atômica (temp + os.replace)."""
    import joblib

    from triage.train import METADATA_FILENAME, MODEL_FILENAME, build_metadata

    candidate_dir, model_dir = Path(candidate_dir), Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    candidate_model = candidate_dir / MODEL_FILENAME
    metadata = build_metadata(joblib.load(candidate_model), metrics, extra)

    artifacts = [p for p in candidate_dir.iterdir() if p.is_file() and p.name != METADATA_FILENAME]
    for source in artifacts:
        _atomic_copy(source, model_dir / source.name)
    _atomic_write(
        model_dir / METADATA_FILENAME,
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
    )
    log.info("modelo promovido: version=%s -> %s", metadata["model_version"], model_dir)
    return {"model_version": metadata["model_version"], "model_dir": str(model_dir)}


def _atomic_copy(source: Path, target: Path) -> None:
    fd, tmp = tempfile.mkstemp(dir=target.parent, suffix=".tmp")
    os.close(fd)
    shutil.copyfile(source, tmp)
    os.chmod(tmp, 0o644)
    os.replace(tmp, target)


def _atomic_write(target: Path, content: str) -> None:
    fd, tmp = tempfile.mkstemp(dir=target.parent, suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(content)
    os.chmod(tmp, 0o644)
    os.replace(tmp, target)


def run_all(settings: Settings, params: dict[str, Any] | None = None) -> dict[str, Any]:
    ingested = ingest(settings.data_dir)
    candidate_dir = settings.model_dir / "candidates" / "cli"
    trained = train(ingested["train_path"], candidate_dir, params)
    metrics = evaluate(trained["model_path"], ingested["test_path"], settings.reports_dir)
    quality_gate(metrics, settings.quality_gate_f1)
    exported = export_onnx(trained["model_path"], ingested["test_path"], candidate_dir)
    promoted = promote(
        candidate_dir,
        settings.model_dir,
        metrics,
        extra={
            "n_train": ingested["n_train"],
            "n_test": ingested["n_test"],
            "onnx": onnx_metadata(exported),
        },
    )
    return {**ingested, **trained, **metrics, **exported, **promoted}


def onnx_metadata(exported: dict[str, Any]) -> dict[str, Any]:
    """Bloco `onnx` do metadata.json (§3.4) a partir do retorno de export_onnx."""
    return {
        "available": True,
        "parity_label_agreement": exported["label_agreement"],
        "max_abs_proba_diff": exported["max_abs_proba_diff"],
        "median_abs_proba_diff": exported["median_abs_proba_diff"],
        "parity_n": exported["n"],
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="triage.pipeline", description="Pipeline de treino")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("ingest")
    p_train = sub.add_parser("train")
    p_train.add_argument("--train-path")
    p_train.add_argument("--candidate-dir")
    p_eval = sub.add_parser("evaluate")
    p_eval.add_argument("--model-path")
    p_eval.add_argument("--test-path")
    p_onnx = sub.add_parser("export-onnx")
    p_onnx.add_argument("--model-path")
    p_onnx.add_argument("--test-path")
    p_onnx.add_argument("--candidate-dir")
    p_promote = sub.add_parser("promote")
    p_promote.add_argument("--candidate-dir")
    sub.add_parser("run-all")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    settings = Settings()
    logging.basicConfig(level=settings.log_level, format="%(levelname)s %(name)s: %(message)s")
    default_candidate = settings.model_dir / "candidates" / "cli"

    match args.command:
        case "ingest":
            result = ingest(settings.data_dir)
        case "train":
            train_path = args.train_path or settings.processed_dir / "train.csv"
            result = train(train_path, args.candidate_dir or default_candidate, DEFAULT_PARAMS)
        case "evaluate":
            model_path = args.model_path or default_candidate / "model.joblib"
            test_path = args.test_path or settings.processed_dir / "test.csv"
            result = evaluate(model_path, test_path, settings.reports_dir)
        case "export-onnx":
            model_path = args.model_path or default_candidate / "model.joblib"
            test_path = args.test_path or settings.processed_dir / "test.csv"
            result = export_onnx(model_path, test_path, args.candidate_dir or default_candidate)
        case "promote":
            metrics_file = settings.reports_dir / "metrics" / "classification_report.json"
            metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
            result = promote(args.candidate_dir or default_candidate, settings.model_dir, metrics)
        case _:
            result = run_all(settings, DEFAULT_PARAMS)

    print(json.dumps({k: v for k, v in result.items() if k != "classification_report"}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
