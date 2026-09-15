# Comparação de latência — sklearn vs. ONNX Runtime

Medido em 2026-09-15T23:50:16Z · modelo `20260915T234937Z`.

| Cenário | Backend | Batch | p50 (ms) | p95 (ms) | p99 (ms) | Throughput (req/s) | Speedup p50 |
|---|---|---|---|---|---|---|---|
| In-process | sklearn | 1 | 0.2194 | 0.2686 | 0.3118 | 4487 | 1.00× |
| In-process | onnx | 1 | 0.0938 | 0.1175 | 0.1309 | 10449 | 2.34× |
| In-process | sklearn | 32 | 2.3782 | 2.6962 | 2.7410 | 13350 | 1.00× |
| In-process | onnx | 32 | 1.5089 | 1.6171 | 1.6511 | 21125 | 1.58× |
| HTTP (Docker) | sklearn | 1 | 1.4037 | 2.0038 | 2.3663 | 673 | 1.00× |
| HTTP (Docker) | onnx | 1 | 1.0446 | 1.6311 | 2.0843 | 865 | 1.34× |

## Paridade

- Concordância de rótulos: **100.00%** em 2888 textos do conjunto de teste.
- Diferença de probabilidade: mediana 3.06e-08, p99,9 8.98e-04, máxima 2.25e-02 (3 textos acima de 1e-3).

## Artefatos

- `model.joblib`: 1.21 MB
- `model.onnx`: 0.80 MB

## Ambiente

- CPU: Apple M4
- Plataforma: macOS-26.6.2-arm64-arm-64bit
- Versões: python 3.12.13, scikit-learn 1.8.0, skl2onnx 1.20.0, onnxruntime 1.30.0, numpy 2.5.3
- Threads: intra_op=1, OMP_NUM_THREADS=1
