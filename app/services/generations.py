"""Generation lifecycle.

``create`` stores the upload and a ``pending`` row, and the request returns right away.
``process`` runs in the background: it marks the row ``processing``, calls inference,
stores the mesh, then marks the row ``completed`` or ``failed``. A semaphore limits how
many jobs reach the GPU at once.
"""

import asyncio
import logging
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.models import Generation, GenerationStatus
from app.schemas import OutputFormat, Quality
from app.services.inference_client import InferenceClient, InferenceError
from app.services.storage import LocalStorage

logger = logging.getLogger(__name__)

MAX_ERROR_LENGTH: Final = 500
IMAGE_CONTENT_TYPES: Final = {"png": "image/png", "jpg": "image/jpeg", "webp": "image/webp"}
GENERIC_FAILURE_MESSAGE: Final = "Mesh generation failed unexpectedly. Please try again."


class GenerationService:
    def __init__(
        self,
        sessionmaker: async_sessionmaker[AsyncSession],
        inference: InferenceClient,
        storage: LocalStorage,
        max_concurrent: int = 1,
    ) -> None:
        self._sessionmaker = sessionmaker
        self._inference = inference
        self._storage = storage
        self._gpu_slots = asyncio.Semaphore(max_concurrent)

    async def create(
        self,
        session: AsyncSession,
        *,
        user_id: uuid.UUID,
        source_filename: str,
        image: bytes,
        extension: str,
        quality: Quality,
        output_format: OutputFormat,
    ) -> Generation:
        generation = Generation(
            user_id=user_id,
            status=GenerationStatus.PENDING,
            quality=quality,
            output_format=output_format,
            source_filename=source_filename[:255],
        )
        session.add(generation)
        await session.flush()  # assigns generation.id

        generation.input_key = f"inputs/{generation.id}.{extension}"
        self._storage.write(generation.input_key, image)
        await session.commit()
        return generation

    async def process(self, generation_id: uuid.UUID) -> None:
        """Background task. It always leaves the row in ``completed`` or ``failed``, never ``processing``."""
        async with self._sessionmaker() as session:
            generation = await session.get(Generation, generation_id)
            if generation is None or generation.status is not GenerationStatus.PENDING:
                return

            generation.status = GenerationStatus.PROCESSING
            await session.commit()

            try:
                await self._run(generation)
            except Exception as exc:
                logger.exception("Generation %s failed", generation.id)
                generation.status = GenerationStatus.FAILED
                generation.error_message = _describe_failure(exc)
            finally:
                if generation.input_key:
                    self._storage.delete(generation.input_key)

            await session.commit()

    async def _run(self, generation: Generation) -> None:
        extension = Path(generation.input_key).suffix.lstrip(".")
        image = self._storage.read(generation.input_key)

        async with self._gpu_slots:
            payload = await self._inference.generate_mesh(
                image=image,
                filename=f"{generation.id}.{extension}",
                content_type=IMAGE_CONTENT_TYPES[extension],
                quality=generation.quality,
                output_format=generation.output_format,
            )

        output_key = f"meshes/{generation.id}.{generation.output_format}"
        self._storage.write(output_key, payload.mesh)

        generation.output_key = output_key
        generation.vertex_count = payload.vertex_count
        generation.face_count = payload.face_count
        generation.completed_at = datetime.now(UTC)
        generation.status = GenerationStatus.COMPLETED


def _describe_failure(exc: Exception) -> str:
    # Expected inference failures carry a user-safe message. Anything else gets a generic one.
    message = str(exc) if isinstance(exc, InferenceError) else GENERIC_FAILURE_MESSAGE
    return message[:MAX_ERROR_LENGTH]
