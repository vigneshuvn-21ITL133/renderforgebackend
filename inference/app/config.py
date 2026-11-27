"""
Centralized configuration for the Renderforge inference service.
All values are overridable via environment variables (see .env.example).
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="RF_", extra="ignore")

    # --- Server ---
    HOST: str = "0.0.0.0"
    PORT: int = 8001
    ALLOWED_ORIGINS: list[str] = ["*"]  # tighten in production (Django backend origin)
    API_KEY: str | None = None  # if set, required via X-API-Key header on /infer

    # --- Engine: one model per process (VRAM) ---
    ENGINE: Literal["triposr", "hunyuan3d_mini"] = "triposr"

    # Hunyuan3D-2mini (shape only). quality -> (inference steps, octree resolution)
    HUNYUAN_MODEL: str = "tencent/Hunyuan3D-2mini"
    HUNYUAN_SUBFOLDER: str = "hunyuan3d-dit-v2-mini"
    HUNYUAN_NUM_CHUNKS: int = 8000  # lower = less VRAM, slower. Upstream example uses 20000
    HUNYUAN_SEED: int = 12345
    HUNYUAN_QUALITY: dict[str, tuple[int, int]] = {
        "low": (20, 256),
        "medium": (30, 320),
        "high": (50, 380),
    }

    # --- Compute ---
    DEVICE: Literal["cuda", "cpu", "auto"] = "auto"
    MODEL_CHUNK_SIZE: int = 8192  # lower (e.g. 4096) if you hit VRAM OOM
    PRETRAINED_MODEL: str = "stabilityai/TripoSR"
    MODEL_CONFIG_NAME: str = "config.yaml"
    MODEL_WEIGHT_NAME: str = "model.ckpt"

    # --- Inference defaults ---
    DEFAULT_MC_RESOLUTION: int = 256          # marching cubes resolution
    QUALITY_RESOLUTIONS: dict[str, int] = {
        "low": 128,
        "medium": 256,
        "high": 320,
    }
    DEFAULT_FOREGROUND_RATIO: float = 0.85
    MAX_UPLOAD_MB: int = 10
    ALLOWED_CONTENT_TYPES: tuple[str, ...] = ("image/png", "image/jpeg", "image/webp")

    # --- Mesh post-processing ---
    ENABLE_MESH_CLEANUP: bool = True
    ENABLE_DECIMATION: bool = True
    DECIMATION_TARGET_FACES: int = 20000  # cap output complexity for web/AR use
    SMOOTH_ITERATIONS: int = 10           # Taubin smoothing passes against marching-cubes bumps; 0 turns it off
    MIN_COMPONENT_RATIO: float = 0.03     # drop floating pieces smaller than this share of the largest piece
    MC_THRESHOLD: float = 25.0            # TripoSR iso-surface level. Lower = thicker/fuller shape, higher = thinner
    REMBG_MODEL: str = "u2net"            # "isnet-general-use" gives cleaner edges on many objects

    # --- Storage (local scratch; permanent storage handled upstream by Django/R2) ---
    OUTPUT_DIR: Path = Path("/tmp/renderforge/outputs")
    JOB_TTL_SECONDS: int = 3600  # how long generated files are kept before cleanup

    def resolved_device(self) -> str:
        if self.DEVICE != "auto":
            return self.DEVICE
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"


settings = Settings()
settings.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
