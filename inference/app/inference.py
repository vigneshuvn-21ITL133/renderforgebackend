"""
The actual image -> mesh pipeline, matching the "AI 3D Generation Pipeline"
stages from the system design:

    Preprocess Image -> Image to 3D (engine) -> Mesh Processing -> Optimization

Preprocessing and the model call live in app/engines.py (TripoSR or
Hunyuan3D-2mini). This module only does the shared mesh cleanup and export.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import trimesh
from PIL import Image

from app.config import settings
from app.schemas import InferParams


@dataclass
class InferenceResult:
    job_id: str
    mesh_path: Path
    output_format: str
    vertex_count: int
    face_count: int
    inference_seconds: float
    device: str
    engine: str


def _cleanup_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Removes degenerate geometry and fixes winding/normals."""
    mesh.remove_infinite_values()
    mesh.merge_vertices()
    mesh.remove_duplicate_faces()
    mesh.remove_unreferenced_vertices()
    mesh.fix_normals()
    return mesh


def _decimate_mesh(mesh: trimesh.Trimesh, target_faces: int) -> trimesh.Trimesh:
    if len(mesh.faces) <= target_faces:
        return mesh
    try:
        return mesh.simplify_quadric_decimation(target_faces)
    except Exception:
        # simplify_quadric_decimation requires the `fast-simplification` extra;
        # degrade gracefully rather than failing the whole job.
        return mesh


def run_inference(image: Image.Image, params: InferParams) -> InferenceResult:
    from app.model import get_device, get_model

    start = time.perf_counter()
    job_id = uuid.uuid4().hex[:12]
    engine = get_model()

    mesh: trimesh.Trimesh = engine.generate(image, params)

    if settings.ENABLE_MESH_CLEANUP:
        mesh = _cleanup_mesh(mesh)
    if settings.ENABLE_DECIMATION:
        mesh = _decimate_mesh(mesh, settings.DECIMATION_TARGET_FACES)

    job_dir = settings.OUTPUT_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    mesh_path = job_dir / f"model.{params.output_format}"
    mesh.export(mesh_path)

    return InferenceResult(
        job_id=job_id,
        mesh_path=mesh_path,
        output_format=params.output_format,
        vertex_count=len(mesh.vertices),
        face_count=len(mesh.faces),
        inference_seconds=round(time.perf_counter() - start, 2),
        device=get_device(),
        engine=engine.name,
    )
