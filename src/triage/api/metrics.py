"""Métricas Prometheus e middleware HTTP (contrato §3.7 do guia).

As métricas são definidas no nível do módulo, uma única vez por processo: criá-las
dentro de create_app() quebraria os testes com "Duplicated timeseries in CollectorRegistry".
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from time import perf_counter

from fastapi import FastAPI, Request, Response
from prometheus_client import Counter, Gauge, Histogram
from starlette.routing import Match

METRICS_PATH = "/metrics"
UNMATCHED = "unmatched"

REQUESTS_TOTAL = Counter(
    "triage_http_requests_total",
    "Requisições HTTP atendidas.",
    ["method", "handler", "status"],
)

REQUEST_DURATION = Histogram(
    "triage_http_request_duration_seconds",
    "Duração das requisições HTTP.",
    ["method", "handler"],
    buckets=(0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5),
)

INFERENCE_DURATION = Histogram(
    "triage_model_inference_duration_seconds",
    "Duração da inferência do modelo, sem o overhead de HTTP.",
    ["backend"],
    buckets=(0.0001, 0.00025, 0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1),
)

PREDICTIONS_TOTAL = Counter(
    "triage_predictions_total",
    "Predições por classe.",
    ["label", "backend"],
)

MODEL_INFO = Gauge(
    "triage_model_info",
    "Modelo carregado (valor sempre 1; a informação está nos labels).",
    ["model_version", "backend"],
)


def observe_inference(backend: str, seconds: float) -> None:
    INFERENCE_DURATION.labels(backend=backend).observe(seconds)


def count_prediction(label: str, backend: str) -> None:
    PREDICTIONS_TOTAL.labels(label=label, backend=backend).inc()


def set_model_info(model_version: str, backend: str) -> None:
    MODEL_INFO.labels(model_version=model_version, backend=backend).set(1)


def resolve_handler(request: Request) -> str:
    """Template da rota (ex.: "/predict"), nunca o path bruto: evita explodir a cardinalidade."""
    for route in request.app.routes:
        match, _ = route.matches(request.scope)
        if match == Match.FULL:
            return route.path
    return UNMATCHED


def register_middleware(app: FastAPI) -> None:
    @app.middleware("http")
    async def instrument(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        if request.url.path == METRICS_PATH:
            return await call_next(request)

        handler = resolve_handler(request)
        method = request.method
        status = "500"
        start = perf_counter()
        try:
            response = await call_next(request)
            status = str(response.status_code)
            return response
        finally:
            REQUEST_DURATION.labels(method=method, handler=handler).observe(perf_counter() - start)
            REQUESTS_TOTAL.labels(method=method, handler=handler, status=status).inc()
