from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class InferParams(BaseModel):
    """Form-encoded alongside the uploaded image on POST /infer."""

    quality: Literal["low", "medium", "high"] = "medium"
    output_format: Literal["glb", "obj"] = "glb"
    remove_background: bool = True
    foreground_ratio: float = Field(0.85, ge=0.5, le=1.0)


class InferResponse(BaseModel):
    job_id: str
    status: Literal["completed"]
    output_format: str
    mesh_url: str
    thumbnail_url: str | None = None
    vertex_count: int
    face_count: int
    inference_seconds: float
    device: str
    engine: str


class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    status: Literal["ok", "loading", "error"]
    model_loaded: bool
    engine: str
    device: str
    cuda_available: bool
    gpu_name: str | None = None
    vram_total_mb: float | None = None
    vram_used_mb: float | None = None


class ErrorResponse(BaseModel):
    detail: str
