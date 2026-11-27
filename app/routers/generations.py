"""Mesh generation endpoints: submit an image, poll status, download the mesh."""

import uuid
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, Query, UploadFile, status
from fastapi.responses import FileResponse
from sqlalchemy import select

from app.deps import CurrentUser, GenerationServiceDep, SessionDep, SettingsDep, StorageDep
from app.images import InvalidImageError, detect_image, read_image_upload
from app.models import Generation, GenerationStatus
from app.schemas import MESH_MEDIA_TYPES, GenerationOut, OutputFormat, Quality

router = APIRouter(prefix="/generations", tags=["generations"])


def _to_out(generation: Generation) -> GenerationOut:
    out = GenerationOut.model_validate(generation)
    if generation.status is GenerationStatus.COMPLETED:
        out.download_url = f"/generations/{generation.id}/download"
    return out


async def _get_owned(session: SessionDep, user_id: uuid.UUID, generation_id: uuid.UUID) -> Generation:
    generation = await session.scalar(
        select(Generation).where(Generation.id == generation_id, Generation.user_id == user_id)
    )
    if generation is None:
        # 404 rather than 403, so other users' generation IDs cannot be probed.
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Generation not found")
    return generation


@router.post("", response_model=GenerationOut, status_code=status.HTTP_202_ACCEPTED)
async def create_generation(
    background: BackgroundTasks,
    user: CurrentUser,
    session: SessionDep,
    settings: SettingsDep,
    service: GenerationServiceDep,
    image: Annotated[UploadFile, File()],
    quality: Annotated[Quality, Form()] = "medium",
    output_format: Annotated[OutputFormat, Form()] = "glb",
) -> GenerationOut:
    data = await read_image_upload(image, settings.max_upload_bytes)
    try:
        extension, _mime = detect_image(data)
    except InvalidImageError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc

    generation = await service.create(
        session,
        user_id=user.id,
        source_filename=image.filename or "upload",
        image=data,
        extension=extension,
        quality=quality,
        output_format=output_format,
    )
    background.add_task(service.process, generation.id)
    return _to_out(generation)


@router.get("", response_model=list[GenerationOut])
async def list_generations(
    user: CurrentUser,
    session: SessionDep,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> list[GenerationOut]:
    rows = await session.scalars(
        select(Generation)
        .where(Generation.user_id == user.id)
        .order_by(Generation.created_at.desc(), Generation.id)
        .limit(limit)
        .offset(offset)
    )
    return [_to_out(generation) for generation in rows]


@router.get("/{generation_id}", response_model=GenerationOut)
async def get_generation(generation_id: uuid.UUID, user: CurrentUser, session: SessionDep) -> GenerationOut:
    return _to_out(await _get_owned(session, user.id, generation_id))


@router.get("/{generation_id}/download")
async def download_generation(
    generation_id: uuid.UUID,
    user: CurrentUser,
    session: SessionDep,
    storage: StorageDep,
) -> FileResponse:
    generation = await _get_owned(session, user.id, generation_id)
    if generation.status is not GenerationStatus.COMPLETED or generation.output_key is None:
        raise HTTPException(status.HTTP_409_CONFLICT, "The mesh is not ready yet")

    path: Path = storage.path(generation.output_key)
    fmt = generation.output_format
    return FileResponse(
        path,
        media_type=MESH_MEDIA_TYPES[fmt],
        filename=f"renderforge-{generation.id}.{fmt}",
        # A finished mesh never changes, so the browser can reuse it when a history item is reopened.
        headers={"Cache-Control": "private, max-age=86400"},
    )
