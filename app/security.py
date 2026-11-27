"""Password hashing and JWT issuing/verification."""

import asyncio
import uuid
from datetime import UTC, datetime, timedelta
from typing import Literal

from jose import JWTError, jwt
from passlib.context import CryptContext

from app.config import Settings

TokenType = Literal["access", "refresh"]

_pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
# Verified against when the email is unknown, so login takes the same time either way.
_DUMMY_HASH = _pwd_context.hash("renderforge-timing-equaliser")


class InvalidTokenError(Exception):
    """The token is malformed, expired, tampered with, or of the wrong type."""


def hash_password(password: str) -> str:
    return _pwd_context.hash(password)


def verify_password(plain: str, hashed: str) -> bool:
    return _pwd_context.verify(plain, hashed)


def burn_password_check(plain: str) -> None:
    _pwd_context.verify(plain, _DUMMY_HASH)


# bcrypt takes ~100-300 ms of pure CPU. Run on a worker thread so one login never stalls every other request.
async def hash_password_async(password: str) -> str:
    return await asyncio.to_thread(hash_password, password)


async def verify_password_async(plain: str, hashed: str) -> bool:
    return await asyncio.to_thread(verify_password, plain, hashed)


async def burn_password_check_async(plain: str) -> None:
    await asyncio.to_thread(burn_password_check, plain)


def create_token(user_id: uuid.UUID, token_type: TokenType, settings: Settings) -> str:
    now = datetime.now(UTC)
    lifetime = (
        timedelta(minutes=settings.access_token_expire_minutes)
        if token_type == "access"
        else timedelta(days=settings.refresh_token_expire_days)
    )
    claims = {
        "sub": str(user_id),
        "type": token_type,
        "iat": int(now.timestamp()),
        "exp": int((now + lifetime).timestamp()),
        "jti": uuid.uuid4().hex,
    }
    return jwt.encode(claims, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)


def decode_token(token: str, expected_type: TokenType, settings: Settings) -> uuid.UUID:
    """Return the user ID in a valid token of the expected type, or raise InvalidTokenError."""
    try:
        claims = jwt.decode(token, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm])
    except JWTError as exc:
        raise InvalidTokenError("Invalid token") from exc

    if claims.get("type") != expected_type or not claims.get("sub"):
        raise InvalidTokenError("Wrong token type")
    try:
        return uuid.UUID(claims["sub"])
    except ValueError as exc:
        raise InvalidTokenError("Malformed subject") from exc
