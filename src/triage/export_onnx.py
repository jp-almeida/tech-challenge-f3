"""Conversão do pipeline sklearn para ONNX e verificação de paridade.

A configuração do conversor é sensível: sem `sublinear_tf` no treino e com `tokenexp`
explícito aqui a concordância de rótulos é de 100%; com `sublinear_tf` ligado ela cai
para 98,75% no conjunto de teste completo.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

ONNX_FILENAME = "model.onnx"

# Mesmo token_pattern do TfidfVectorizer do sklearn (r"(?u)\b\w\w+\b").
TOKEN_EXPRESSION = r"\b\w\w+\b"

CONVERTER_OPTIONS: dict[Any, dict[str, Any]] = {
    TfidfVectorizer: {"tokenexp": TOKEN_EXPRESSION},
    # zipmap=False: probabilidades como tensor (N, 3) em vez de lista de dicts.
    LogisticRegression: {"zipmap": False},
}

# O que o gate cobra, e por quê.
#
# O critério que importa é o rótulo: se sklearn e ONNX decidem igual, a troca de backend é
# transparente para quem consome a API.
MIN_LABEL_AGREEMENT = 0.995

# A mediana pega divergência sistemática — foi ela que reprovou a configuração com
# `sublinear_tf=True` (mediana ~5e-3, quatro ordens acima do normal, que é ~3e-8).
MAX_MEDIAN_PROBA_DIFF = 1e-4

# O p99,9 é só uma rede de segurança para a cauda, com folga deliberada. Duas fontes de
# divergência pontual são conhecidas e benignas:
#   1. o TfIdfVectorizer do ONNX monta n-gramas a partir do pool de unigramas e descarta
#      bigramas cujo componente foi podado por min_df (aqui, "von hippel");
#   2. a ordem de acumulação em float32 difere entre o BLAS do sklearn e os kernels do
#      ONNX Runtime, e o quanto ela difere depende da arquitetura da CPU.
# Por isso o limiar não é ajustado aos números de uma máquina: medimos p99,9 de 9e-4 em
# Apple Silicon e 2,8e-3 em x86, ambos sem trocar nenhum rótulo.
MAX_P999_PROBA_DIFF = 1e-2


def convert_to_onnx(pipeline: Pipeline) -> bytes:
    # skl2onnx é dependência de treino (requirements/train.txt): importar aqui dentro
    # mantém a imagem da API livre do conversor, que ela não usa para inferir.
    from skl2onnx import get_latest_tested_opset_version, to_onnx
    from skl2onnx.common.data_types import StringTensorType

    model = to_onnx(
        pipeline,
        initial_types=[("text", StringTensorType([None, 1]))],
        options=CONVERTER_OPTIONS,
        # Opset testado pelo skl2onnx instalado; o onnxruntime suporta versões >= esta.
        target_opset=get_latest_tested_opset_version(),
    )
    return model.SerializeToString()


def make_session(onnx_path: str | Path, threads: int = 1):
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    return ort.InferenceSession(str(onnx_path), options, providers=["CPUExecutionProvider"])


def proba_output_name(session, n_classes: int = 3) -> str:
    """Saída de probabilidades = a de shape [N, n_classes]; os nomes não são hardcoded."""
    for output in session.get_outputs():
        if len(output.shape) == 2 and output.shape[1] in (n_classes, None):
            return output.name
    raise ValueError(f"saída de probabilidades não encontrada em {session.get_outputs()}")


def run_onnx(session, texts: list[str]) -> np.ndarray:
    input_name = session.get_inputs()[0].name
    batch = np.array(texts, dtype=object).reshape(-1, 1)
    proba = session.run([proba_output_name(session)], {input_name: batch})[0]
    return np.asarray(proba, dtype=np.float64)


def check_parity(pipeline: Pipeline, onnx_path: str | Path, texts: list[str]) -> dict[str, Any]:
    """Compara sklearn e ONNX sobre `texts` e falha se a divergência passar do aceitável."""
    session = make_session(onnx_path)
    onnx_proba = run_onnx(session, texts)
    sklearn_proba = pipeline.predict_proba(texts)

    diffs = np.abs(onnx_proba - sklearn_proba).max(axis=1)
    agreement = float((onnx_proba.argmax(axis=1) == sklearn_proba.argmax(axis=1)).mean())
    report = {
        "n": len(texts),
        "label_agreement": agreement,
        "max_abs_proba_diff": float(diffs.max()),
        "median_abs_proba_diff": float(np.median(diffs)),
        "p999_abs_proba_diff": float(np.quantile(diffs, 0.999)),
        "n_above_1e_3": int((diffs > 1e-3).sum()),
    }

    problems = []
    if agreement < MIN_LABEL_AGREEMENT:
        problems.append(f"label_agreement {agreement:.4f} < {MIN_LABEL_AGREEMENT}")
    if report["median_abs_proba_diff"] > MAX_MEDIAN_PROBA_DIFF:
        median = report["median_abs_proba_diff"]
        problems.append(f"mediana da diferença {median:.2e} > {MAX_MEDIAN_PROBA_DIFF:.0e}")
    if report["p999_abs_proba_diff"] > MAX_P999_PROBA_DIFF:
        problems.append(
            f"p99,9 da diferença {report['p999_abs_proba_diff']:.2e} > {MAX_P999_PROBA_DIFF:.0e}"
        )
    if problems:
        worst = np.argsort(diffs)[-5:][::-1]
        details = [{"index": int(i), "diff": float(diffs[i])} for i in worst]
        raise ParityError(
            "paridade sklearn/ONNX insuficiente: "
            + "; ".join(problems)
            + f"\npiores casos: {json.dumps(details)}"
        )
    return report


class ParityError(RuntimeError):
    """Divergência entre sklearn e ONNX acima do tolerado."""
