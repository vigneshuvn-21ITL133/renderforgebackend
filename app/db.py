"""Database engine, session factory and declarative base.

Works with a local Postgres, SQLite (tests) and Neon. Neon hands out URLs like
``postgresql://user:pw@ep-xxx-pooler.region.aws.neon.tech/db?sslmode=require&channel_binding=require``.
asyncpg understands neither the scheme nor those query options, so ``prepare_database_url``
rewrites them into the form SQLAlchemy and asyncpg accept.
"""

from typing import Any
from urllib.parse import parse_qsl, urlencode

from sqlalchemy import update
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

# libpq options asyncpg rejects. SSL is switched on through connect_args instead.
_LIBPQ_ONLY_OPTIONS = {"sslmode", "channel_binding", "options", "sslrootcert", "sslcert", "sslkey", "target_session_attrs"}
_SSL_DISABLED = {"disable", "allow"}


class Base(DeclarativeBase):
    pass


def prepare_database_url(url: str) -> tuple[str, dict[str, Any]]:
    """Return ``(sqlalchemy_url, connect_args)`` for any Postgres, Neon or SQLite URL."""
    parsed = make_url(url)
    if parsed.get_backend_name() not in ("postgres", "postgresql"):
        return url, {}

    if parsed.drivername in ("postgres", "postgresql", "postgresql+psycopg2", "postgres+asyncpg"):
        parsed = parsed.set(drivername="postgresql+asyncpg")

    query = dict(parsed.query)
    sslmode = str(query.get("sslmode", "")).lower()
    host = parsed.host or ""
    connect_args: dict[str, Any] = {}

    if sslmode not in _SSL_DISABLED and (sslmode or "neon.tech" in host or query.get("ssl") == "require"):
        connect_args["ssl"] = True

    # PgBouncer (Neon's "-pooler" hosts) cannot reuse prepared statements across connections.
    if "-pooler" in host:
        connect_args["statement_cache_size"] = 0
        connect_args["prepared_statement_cache_size"] = 0

    cleaned = {key: value for key, value in query.items() if key not in _LIBPQ_ONLY_OPTIONS and key != "ssl"}
    parsed = parsed.set(query=cleaned)
    return parsed.render_as_string(hide_password=False), connect_args


class Database:
    def __init__(
        self, url: str, *, pool_size: int = 5, max_overflow: int = 5, pool_recycle_seconds: int = 240
    ) -> None:
        sa_url, connect_args = prepare_database_url(url)
        kwargs: dict[str, Any] = {"pool_pre_ping": True}
        if sa_url.startswith(("postgresql", "postgres")):
            # Neon suspends idle compute and drops idle connections, so recycle before that happens.
            kwargs.update(pool_recycle=pool_recycle_seconds, pool_size=pool_size, max_overflow=max_overflow)
            connect_args.setdefault("timeout", 30)  # first connect can wake a suspended compute
        self.engine: AsyncEngine = create_async_engine(sa_url, connect_args=connect_args, **kwargs)
        self.sessionmaker = async_sessionmaker(self.engine, expire_on_commit=False)

    async def create_all(self) -> None:
        async with self.engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)

    async def check(self) -> None:
        """Raise if the database cannot be reached."""
        async with self.engine.connect() as connection:
            await connection.exec_driver_sql("SELECT 1")

    async def fail_interrupted_generations(self) -> int:
        """Mark jobs left pending/processing by a restart as failed. Their upload is gone from memory."""
        from app.models import Generation, GenerationStatus

        async with self.sessionmaker() as session:
            result = await session.execute(
                update(Generation)
                .where(Generation.status.in_([GenerationStatus.PENDING, GenerationStatus.PROCESSING]))
                .values(
                    status=GenerationStatus.FAILED,
                    error_message="The server restarted while this model was being made. Please upload the image again.",
                )
            )
            await session.commit()
            return result.rowcount or 0

    async def dispose(self) -> None:
        await self.engine.dispose()
