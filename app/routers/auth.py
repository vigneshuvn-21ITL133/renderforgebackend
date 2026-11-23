"""Account signup, login, token refresh and the current-user endpoint."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from app.config import Settings
from app.deps import CurrentUser, SessionDep, SettingsDep
from app.models import User
from app.schemas import RefreshRequest, SignupRequest, TokenPair, UserOut
from app.security import (
    InvalidTokenError,
    burn_password_check,
    create_token,
    decode_token,
    hash_password,
    verify_password,
)

router = APIRouter(prefix="/auth", tags=["auth"])


def _issue_tokens(user: User, settings: Settings) -> TokenPair:
    return TokenPair(
        access_token=create_token(user.id, "access", settings),
        refresh_token=create_token(user.id, "refresh", settings),
    )


def _invalid_credentials() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Incorrect email or password",
        headers={"WWW-Authenticate": "Bearer"},
    )


@router.post("/signup", response_model=TokenPair, status_code=status.HTTP_201_CREATED)
async def signup(body: SignupRequest, session: SessionDep, settings: SettingsDep) -> TokenPair:
    email = body.email.lower()
    user = User(
        email=email,
        hashed_password=hash_password(body.password),
        full_name=body.full_name,
    )
    session.add(user)
    try:
        await session.commit()
    except IntegrityError as exc:
        # The unique index catches concurrent signups that the check below would miss.
        await session.rollback()
        raise HTTPException(status.HTTP_409_CONFLICT, "An account with this email already exists") from exc
    return _issue_tokens(user, settings)


@router.post("/login", response_model=TokenPair)
async def login(
    form: Annotated[OAuth2PasswordRequestForm, Depends()],
    session: SessionDep,
    settings: SettingsDep,
) -> TokenPair:
    user = await session.scalar(select(User).where(User.email == form.username.lower()))
    if user is None:
        burn_password_check(form.password)
        raise _invalid_credentials()
    if not verify_password(form.password, user.hashed_password):
        raise _invalid_credentials()
    if not user.is_active:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "This account is disabled")
    return _issue_tokens(user, settings)


@router.post("/refresh", response_model=TokenPair)
async def refresh(body: RefreshRequest, session: SessionDep, settings: SettingsDep) -> TokenPair:
    """Exchange a refresh token for a new pair. The old refresh token stays valid until it expires."""
    try:
        user_id = decode_token(body.refresh_token, "refresh", settings)
    except InvalidTokenError as exc:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid refresh token") from exc

    user = await session.get(User, user_id)
    if user is None or not user.is_active:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid refresh token")
    return _issue_tokens(user, settings)


@router.get("/me", response_model=UserOut)
async def me(user: CurrentUser) -> User:
    return user
