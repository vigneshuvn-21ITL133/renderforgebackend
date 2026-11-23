"""
Holds the single loaded engine (TripoSR or Hunyuan3D-2mini, chosen by RF_ENGINE)
as a process-wide singleton, so weights go onto the GPU once at startup and
never per request.
"""
from __future__ import annotations

import logging

from app.config import settings
from app.engines import Engine, create_engine

logger = logging.getLogger("renderforge.model")

_engine: Engine | None = None
_device: str | None = None


def load_model() -> Engine:
    """Loads the configured engine. Safe to call repeatedly."""
    global _engine, _device

    if _engine is not None:
        return _engine

    device = settings.resolved_device()
    engine = create_engine(settings.ENGINE)
    engine.load(device)

    _engine, _device = engine, device
    logger.info("Engine '%s' ready on %s", engine.name, device)
    return _engine


def get_model() -> Engine:
    if _engine is None:
        raise RuntimeError("Model not loaded yet, call load_model() at startup.")
    return _engine


def get_device() -> str:
    if _device is None:
        raise RuntimeError("Model not loaded yet, call load_model() at startup.")
    return _device


def is_loaded() -> bool:
    return _engine is not None


def gpu_stats() -> dict:
    empty = {"cuda_available": False, "gpu_name": None, "vram_total_mb": None, "vram_used_mb": None}
    try:
        import torch
    except ImportError:
        return empty
    if not torch.cuda.is_available():
        return empty
    idx = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(idx)
    return {
        "cuda_available": True,
        "gpu_name": props.name,
        "vram_total_mb": round(props.total_memory / (1024 ** 2), 1),
        "vram_used_mb": round(torch.cuda.memory_allocated(idx) / (1024 ** 2), 1),
    }
