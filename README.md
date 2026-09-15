# Triagem Automática de Laudos — Tech Challenge Fase 3 (MLET)

[![CI](https://github.com/jp-almeida/tech-challenge-f3/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/jp-almeida/tech-challenge-f3/actions/workflows/ci.yml)

> Classificação de laudos em `normal` / `atencao` / `urgente`, servida por API em container,
> com CI/CD, retreino orquestrado (Airflow), monitoramento (Prometheus + Grafana) e
> otimização de latência (ONNX Runtime).

**Status:** em implementação. Ver [docs/GUIA_IMPLEMENTACAO.md](docs/GUIA_IMPLEMENTACAO.md).

## Integrantes

<!-- Etapa 8 -->

## Vídeo (STAR, ≤ 5 min)

<!-- Etapa 9 -->

## Arquitetura

<!-- Etapa 8 -->

## Dataset e mapeamento de rótulos

<!-- Etapa 8 -->

## Quickstart

<!-- Etapa 8 -->

## API

<!-- Etapa 8 -->

## CI/CD

<!-- Etapa 8 -->

## Orquestração (Airflow)

<!-- Etapa 8 -->

## Monitoramento (Prometheus + Grafana)

```bash
make up          # API + Prometheus + Grafana
make load-test   # ~2 min de tráfego, incluindo erros propositais
```

| Serviço | URL | Acesso |
|---|---|---|
| API (Swagger) | http://localhost:8000/docs | — |
| Métricas da API | http://localhost:8000/metrics | — |
| Prometheus | http://localhost:9090/targets | — |
| Grafana | http://localhost:3000 | `admin` / `admin` (anônimo como Viewer) |

O dashboard **"Triagem de Laudos — API"** (pasta *Triagem*) é provisionado automaticamente a
partir de [monitoring/grafana/dashboards/triage-api.json](monitoring/grafana/dashboards/triage-api.json),
sem configuração manual. Painéis: total de requisições, requisições por segundo por endpoint,
latência HTTP p50/p95/p99 de `/predict`, taxa de erro 4xx/5xx, latência de inferência do modelo
por backend e distribuição das predições por classe.

![Dashboard do Grafana](docs/images/grafana-dashboard.png)

## Otimização e comparação de latência

<!-- Etapa 8 -->

## Decisão arquitetural em nuvem (batch vs. real-time)

<!-- Etapa 8 -->

## Limitações

<!-- Etapa 8 -->
