"""
The actual image -> mesh pipeline, matching the "AI 3D Generation Pipeline"
stages from the system design:

    Preprocess Image -> Image to 3D (engine) -> Mesh Processing -> Optimization

Preprocessing and the model call live in app/engines.py (TripoSR or
Hunyuan3D-2mini). This module only does the shared mesh cleanup and export.
"""
from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import trimesh
from PIL import Image
from scipy.spatial import cKDTree

from app.config import settings
from app.schemas import InferParams

logger = logging.getLogger("renderforge.inference")


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


def _remove_small_components(mesh: trimesh.Trimesh, min_ratio: float) -> trimesh.Trimesh:
    """Drop floating specks: keep pieces with at least ``min_ratio`` of the largest piece's faces."""
    if min_ratio <= 0 or len(mesh.faces) == 0:
        return mesh
    parts = mesh.split(only_watertight=False)
    if len(parts) <= 1:
        return mesh
    largest = max(len(part.faces) for part in parts)
    kept = [part for part in parts if len(part.faces) >= largest * min_ratio]
    if len(kept) == len(parts):
        return mesh
    logger.info("Removed %d small floating pieces", len(parts) - len(kept))
    return trimesh.util.concatenate(kept)


def _smooth_mesh(mesh: trimesh.Trimesh, iterations: int) -> trimesh.Trimesh:
    """Taubin smoothing evens out marching-cubes bumps without shrinking the shape. Colours are kept."""
    if iterations > 0 and len(mesh.faces) > 0:
        trimesh.smoothing.filter_taubin(mesh, iterations=iterations)
    return mesh


def _decimate_mesh(mesh: trimesh.Trimesh, target_faces: int) -> trimesh.Trimesh:
    """Reduce to about ``target_faces`` triangles and carry the vertex colours over."""
    if len(mesh.faces) <= target_faces:
        return mesh
    try:
        # face_count must be a keyword: the first positional argument is a percentage.
        simplified = mesh.simplify_quadric_decimation(face_count=target_faces)
    except Exception:
        # Needs the `fast-simplification` package. Keep the full mesh rather than failing the job.
        logger.warning("Decimation skipped", exc_info=True)
        return mesh

    if mesh.visual.kind == "vertex":
        # Simplification drops colours, so copy each new vertex's colour from the nearest original vertex.
        _, nearest = cKDTree(mesh.vertices).query(simplified.vertices)
        simplified.visual = trimesh.visual.ColorVisuals(
            simplified, vertex_colors=mesh.visual.vertex_colors[nearest]
        )
    return simplified


def run_inference(image: Image.Image, params: InferParams) -> InferenceResult:
    from app.model import get_device, get_model

    start = time.perf_counter()
    job_id = uuid.uuid4().hex[:12]
    engine = get_model()

    mesh: trimesh.Trimesh = engine.generate(image, params)

    if settings.ENABLE_MESH_CLEANUP:
        mesh = _cleanup_mesh(mesh)
        mesh = _remove_small_components(mesh, settings.MIN_COMPONENT_RATIO)
        mesh = _smooth_mesh(mesh, settings.SMOOTH_ITERATIONS)
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
