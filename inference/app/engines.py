"""
Image-to-3D engines. Exactly one engine is loaded per server process (chosen by
RF_ENGINE), because two models will not fit in 4-6 GB of VRAM together.

    triposr         fast, small, vertex-colored mesh, rougher geometry
    hunyuan3d_mini  Hunyuan3D-2mini shape model, cleaner geometry, NO colors
                    (shape only, so the GLB is untextured grey)

Each engine exposes:
    load(device)                       -> loads weights once at startup
    generate(image, params)            -> trimesh.Trimesh
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import numpy as np
import trimesh
from PIL import Image

from app.config import settings
from app.schemas import InferParams

logger = logging.getLogger("renderforge.engines")


def _remove_background_rgba(image: Image.Image) -> Image.Image:
    """Returns an RGBA image with the background made transparent (rembg)."""
    import rembg

    return rembg.remove(image.convert("RGBA"), session=_rembg_session())


_session = None


def _rembg_session():
    global _session
    if _session is None:
        import rembg

        _session = rembg.new_session()
    return _session


class Engine(ABC):
    name: str

    @abstractmethod
    def load(self, device: str) -> None: ...

    @abstractmethod
    def generate(self, image: Image.Image, params: InferParams) -> trimesh.Trimesh: ...


class TripoSREngine(Engine):
    name = "triposr"

    def load(self, device: str) -> None:
        from tsr.system import TSR  # installed from source, see README

        logger.info("Loading TripoSR (%s) onto %s", settings.PRETRAINED_MODEL, device)
        model = TSR.from_pretrained(
            settings.PRETRAINED_MODEL,
            config_name=settings.MODEL_CONFIG_NAME,
            weight_name=settings.MODEL_WEIGHT_NAME,
        )
        model.renderer.set_chunk_size(settings.MODEL_CHUNK_SIZE)
        model.to(device)
        self._model = model
        self._device = device

    def _preprocess(self, image: Image.Image, params: InferParams) -> Image.Image:
        from tsr.utils import remove_background, resize_foreground

        if not params.remove_background:
            return image.convert("RGB")

        image = remove_background(image.convert("RGBA"), _rembg_session())
        image = resize_foreground(image, params.foreground_ratio)
        arr = np.array(image).astype(np.float32) / 255.0
        rgb, alpha = arr[:, :, :3], arr[:, :, 3:4]
        # composite onto neutral grey so the model does not see transparency artifacts
        return Image.fromarray(((rgb * alpha + 0.5 * (1 - alpha)) * 255.0).astype(np.uint8))

    def generate(self, image: Image.Image, params: InferParams) -> trimesh.Trimesh:
        import torch

        processed = self._preprocess(image, params)
        resolution = settings.QUALITY_RESOLUTIONS.get(params.quality, settings.DEFAULT_MC_RESOLUTION)
        with torch.no_grad():
            scene_codes = self._model([processed], device=self._device)
            return self._model.extract_mesh(scene_codes, resolution=resolution)[0]


class Hunyuan3DMiniEngine(Engine):
    name = "hunyuan3d_mini"

    def load(self, device: str) -> None:
        # Package name comes from https://github.com/Tencent/Hunyuan3D-2 (pip install -e .)
        from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

        logger.info("Loading %s onto %s", settings.HUNYUAN_MODEL, device)
        self._pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            settings.HUNYUAN_MODEL,
            subfolder=settings.HUNYUAN_SUBFOLDER,
            use_safetensors=True,
            device=device,
        )
        self._device = device

    def generate(self, image: Image.Image, params: InferParams) -> trimesh.Trimesh:
        import torch

        # The pipeline reads the alpha channel, so hand it RGBA with a clean cutout.
        processed = _remove_background_rgba(image) if params.remove_background else image.convert("RGBA")
        steps, octree = settings.HUNYUAN_QUALITY.get(params.quality, settings.HUNYUAN_QUALITY["medium"])

        with torch.no_grad():
            result = self._pipe(
                image=processed,
                num_inference_steps=steps,
                octree_resolution=octree,
                num_chunks=settings.HUNYUAN_NUM_CHUNKS,
                generator=torch.manual_seed(settings.HUNYUAN_SEED),
                output_type="trimesh",
            )
        return result[0]


_REGISTRY: dict[str, type[Engine]] = {
    "triposr": TripoSREngine,
    "hunyuan3d_mini": Hunyuan3DMiniEngine,
}


def create_engine(name: str) -> Engine:
    try:
        return _REGISTRY[name]()
    except KeyError:
        raise ValueError(f"Unknown engine '{name}'. Choose one of: {', '.join(_REGISTRY)}") from None
