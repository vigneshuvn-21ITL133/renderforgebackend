"""
Renderforge Inference Service
==============================
Standalone FastAPI wrapper around TripoSR. This is the "GPU Worker" / "AI 3D
Generation Pipeline" box in the system architecture — it does NOT handle auth,
billing, job queueing, or permanent storage. Celery workers call this service
over HTTP for the actual image->mesh compute, then take the returned file and
upload it to R2/S3 themselves.

Run locally:
    uvicorn app.main:app --host 0.0.0.0 --port 8001

See README.md for full setup (TripoSR install, model download, GPU notes).
"""
from __future__ import annotations

import logging
import shutil
import time
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from PIL import Image, UnidentifiedImageError

from app.config import settings
from app.inference import run_inference
from app.model import gpu_stats, is_loaded, load_model
from app.schemas import ErrorResponse, HealthResponse, InferParams, InferResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("renderforge.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up — loading TripoSR (this can take 10-60s)...")
    load_model()
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="Renderforge Inference Service",
    description="Image-to-3D generation via TripoSR",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


def require_api_key(x_api_key: str | None = Header(default=None)):
    if settings.API_KEY and x_api_key != settings.API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key header")


@app.get("/", include_in_schema=False)
def root():
    return {"service": "renderforge-inference", "status": "ok", "docs": "/docs"}


@app.get("/health", response_model=HealthResponse)
def health():
    stats = gpu_stats()
    return HealthResponse(
        status="ok" if is_loaded() else "loading",
        model_loaded=is_loaded(),
        engine=settings.ENGINE,
        device=settings.resolved_device(),
        cuda_available=stats["cuda_available"],
        gpu_name=stats["gpu_name"],
        vram_total_mb=stats["vram_total_mb"],
        vram_used_mb=stats["vram_used_mb"],
    )


@app.post(
    "/infer",
    response_model=InferResponse,
    responses={400: {"model": ErrorResponse}, 401: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    dependencies=[Depends(require_api_key)],
)
async def infer(
    image: UploadFile = File(..., description="Source image (PNG/JPG/WEBP, max 10MB)"),
    quality: str = Form("medium", description="low | medium | high"),
    output_format: str = Form("glb", description="glb | obj"),
    remove_background: bool = Form(True),
    foreground_ratio: float = Form(0.85),
):
    if image.content_type not in settings.ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=400, detail=f"Unsupported content type: {image.content_type}")

    raw = await image.read()
    if len(raw) > settings.MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(status_code=400, detail=f"Image exceeds {settings.MAX_UPLOAD_MB}MB limit")

    try:
        pil_image = Image.open(__import__("io").BytesIO(raw))
        pil_image.load()
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Could not decode image file")

    try:
        params = InferParams(
            quality=quality,
            output_format=output_format,
            remove_background=remove_background,
            foreground_ratio=foreground_ratio,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    try:
        result = run_inference(pil_image, params)
    except Exception as exc:
        logger.exception("Inference failed")
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}")

    return InferResponse(
        job_id=result.job_id,
        status="completed",
        output_format=result.output_format,
        mesh_url=f"/download/{result.job_id}",
        thumbnail_url=None,
        vertex_count=result.vertex_count,
        face_count=result.face_count,
        inference_seconds=result.inference_seconds,
        device=result.device,
        engine=result.engine,
    )


@app.get("/download/{job_id}", dependencies=[Depends(require_api_key)])
def download(job_id: str):
    job_dir = settings.OUTPUT_DIR / job_id
    if not job_dir.exists():
        raise HTTPException(status_code=404, detail="Job not found or already expired")

    files = list(job_dir.glob("model.*"))
    if not files:
        raise HTTPException(status_code=404, detail="Output file missing")

    mesh_path = files[0]
    return FileResponse(mesh_path, filename=mesh_path.name, media_type="application/octet-stream")


@app.delete("/jobs/{job_id}", dependencies=[Depends(require_api_key)])
def delete_job(job_id: str):
    """Called by the Celery worker once it has copied the file to R2/S3."""
    job_dir = settings.OUTPUT_DIR / job_id
    if job_dir.exists():
        shutil.rmtree(job_dir)
        return {"deleted": True}
    return {"deleted": False}
