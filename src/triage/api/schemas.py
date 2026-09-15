"""Schemas Pydantic do contrato da API (§3.5 do guia)."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from triage.config import MAX_TEXT_CHARS

EXAMPLE_TEXT = (
    "Acute ischemic stroke with left hemiparesis. Patient admitted within three hours of "
    "symptom onset; CT angiography showed occlusion of the right middle cerebral artery."
)

LaudoText = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=10, max_length=MAX_TEXT_CHARS)
]
LabelName = Literal["normal", "atencao", "urgente"]


class PredictRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={"examples": [{"text": EXAMPLE_TEXT}]})

    text: LaudoText = Field(description="Texto do laudo (não é registrado em log).")


class PredictResponse(BaseModel):
    model_config = ConfigDict(
        protected_namespaces=(),
        json_schema_extra={
            "examples": [
                {
                    "label": "urgente",
                    "label_id": 2,
                    "probabilities": {"normal": 0.08, "atencao": 0.17, "urgente": 0.75},
                    "model_version": "20260916T120000Z",
                    "backend": "sklearn",
                    "inference_ms": 0.41,
                }
            ]
        },
    )

    label: LabelName
    label_id: int = Field(ge=0, le=2)
    probabilities: dict[LabelName, float]
    model_version: str
    backend: str
    inference_ms: float


class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    status: Literal["ok", "unavailable"]
    model_loaded: bool
    model_version: str | None = None
    backend: str | None = None


class ModelInfo(BaseModel):
    model_config = ConfigDict(extra="allow", protected_namespaces=())

    model_version: str
    labels: list[str]
    metrics: dict[str, Any]
