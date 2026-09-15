"""Aplicação FastAPI do serviço de triagem.

Toda a lógica fica em create_app(settings), para ser testável; o módulo expõe `app`
para o uvicorn. O texto do laudo nunca é registrado em log (LGPD).
"""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from triage.api.predictor import Predictor, load_predictor
from triage.api.schemas import (
    EXAMPLE_TEXT,
    HealthResponse,
    ModelInfo,
    PredictRequest,
    PredictResponse,
)
from triage.config import LABELS, Settings
from triage.data import normalize_text

log = logging.getLogger("triage.api")


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    logging.basicConfig(level=settings.log_level, format="%(levelname)s %(name)s: %(message)s")

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.predictor = None
        try:
            predictor = load_predictor(settings.backend, settings.model_dir)
            predictor.predict_proba([normalize_text(EXAMPLE_TEXT)])  # warm-up
            app.state.predictor = predictor
            log.info(
                "modelo carregado: version=%s backend=%s dir=%s",
                predictor.model_version,
                predictor.backend,
                settings.model_dir,
            )
        except Exception:
            log.exception("falha ao carregar o modelo de %s", settings.model_dir)
        yield

    app = FastAPI(
        title="Triagem de Laudos",
        description="Classifica laudos em normal / atencao / urgente (apoio à decisão).",
        version="1.0.0",
        lifespan=lifespan,
    )

    def get_predictor(request: Request) -> Predictor:
        predictor = request.app.state.predictor
        if predictor is None:
            raise HTTPException(status_code=503, detail="modelo não carregado")
        return predictor

    @app.get("/health", response_model=HealthResponse, responses={503: {"model": HealthResponse}})
    def health(request: Request):
        predictor = request.app.state.predictor
        if predictor is None:
            body = HealthResponse(status="unavailable", model_loaded=False)
            return JSONResponse(status_code=503, content=body.model_dump())
        return HealthResponse(
            status="ok",
            model_loaded=True,
            model_version=predictor.model_version,
            backend=predictor.backend,
        )

    @app.post("/predict", response_model=PredictResponse)
    def predict(payload: PredictRequest, request: Request) -> PredictResponse:
        predictor = get_predictor(request)
        text = normalize_text(payload.text)

        start = time.perf_counter()
        proba = predictor.predict_proba([text])[0]
        inference_ms = (time.perf_counter() - start) * 1000

        label_id = int(np.argmax(proba))
        return PredictResponse(
            label=LABELS[label_id],
            label_id=label_id,
            probabilities={name: float(p) for name, p in zip(LABELS, proba, strict=True)},
            model_version=predictor.model_version,
            backend=predictor.backend,
            inference_ms=round(inference_ms, 4),
        )

    @app.get("/model/info", response_model=ModelInfo)
    def model_info(request: Request) -> dict:
        return get_predictor(request).metadata

    return app


app = create_app()
