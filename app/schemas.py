"""Request and response models for the public API."""

import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator

from app.models import GenerationStatus

Quality = Literal["low", "medium", "high"]
OutputFormat = Literal["glb", "obj"]

MESH_MEDIA_TYPES = {"glb": "model/gltf-binary", "obj": "model/obj"}


class SignupRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=72, description="bcrypt only uses the first 72 bytes")
    full_name: str | None = Field(default=None, max_length=120)

    @field_validator("password")
    @classmethod
    def _password_fits_bcrypt(cls, value: str) -> str:
        if len(value.encode("utf-8")) > 72:
            raise ValueError("Password must be at most 72 bytes")
        return value


class RefreshRequest(BaseModel):
    refresh_token: str


class TokenPair(BaseModel):
    access_token: str
    refresh_token: str
    token_type: Literal["bearer"] = "bearer"


class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    email: str
    full_name: str | None
    created_at: datetime


class GenerationOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    status: GenerationStatus
    quality: Quality
    output_format: OutputFormat
    source_filename: str
    vertex_count: int | None
    face_count: int | None
    error_message: str | None
    created_at: datetime
    completed_at: datetime | None
    download_url: str | None = None
