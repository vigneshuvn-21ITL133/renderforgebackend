"""FastAPI dependencies: settings, DB session, storage, services and the authenticated user."""

import uuid
from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings
from app.models import User
from app.security import InvalidTokenError, decode_token
from app.services.generations import GenerationService
from app.services.storage import LocalStorage

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")


def get_app_settings(request: Request) -> Settings:
    return request.app.state.settings


async def get_session(request: Request) -> AsyncIterator[AsyncSession]:
    async with request.app.state.database.sessionmaker() as session:
        yield session


def get_storage(request: Request) -> LocalStorage:
    return request.app.state.storage


def get_generation_service(request: Request) -> GenerationService:
    return request.app.state.generations


SettingsDep = Annotated[Settings, Depends(get_app_settings)]
SessionDep = Annotated[AsyncSession, Depends(get_session)]
StorageDep = Annotated[LocalStorage, Depends(get_storage)]
GenerationServiceDep = Annotated[GenerationService, Depends(get_generation_service)]


async def get_current_user(
    token: Annotated[str, Depends(oauth2_scheme)],
    settings: SettingsDep,
    session: SessionDep,
) -> User:
    try:
        user_id = decode_token(token, "access", settings)
    except InvalidTokenError as exc:
        raise _unauthorized() from exc

    user = await session.get(User, user_id)
    if user is None or not user.is_active:
        raise _unauthorized()
    return user


def _unauthorized() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )


CurrentUser = Annotated[User, Depends(get_current_user)]
