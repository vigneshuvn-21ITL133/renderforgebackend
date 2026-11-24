import pytest

from app.db import Database, prepare_database_url
from app.models import Generation, GenerationStatus
from tests.conftest import make_png, signup


NEON = "postgresql://u:p%40ss@ep-cool-123-pooler.ap-southeast-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require"


def test_neon_url_is_rewritten_for_asyncpg():
    url, args = prepare_database_url(NEON)
    assert url.startswith("postgresql+asyncpg://u:p%40ss@ep-cool-123-pooler.")
    assert "sslmode" not in url and "channel_binding" not in url
    assert args["ssl"] is True
    assert args["statement_cache_size"] == 0  # PgBouncer pooler


def test_direct_neon_host_keeps_prepared_statements():
    url, args = prepare_database_url("postgres://u:p@ep-x.eu-central-1.aws.neon.tech/db")
    assert url.startswith("postgresql+asyncpg://")
    assert args == {"ssl": True}


def test_local_postgres_has_no_ssl():
    url, args = prepare_database_url("postgresql+asyncpg://renderforge:pw@localhost:5432/renderforge")
    assert args == {}


def test_sqlite_untouched():
    url = "sqlite+aiosqlite:///x.db"
    assert prepare_database_url(url) == (url, {})


async def test_db_health(client):
    assert (await client.get("/health/db")).json() == {"status": "ok"}


async def test_interrupted_jobs_fail_on_startup(app, client):
    headers = await signup(client)
    created = await client.post("/generations", headers=headers, files={"image": ("a.png", make_png(), "image/png")})
    gid = created.json()["id"]
    db: Database = app.state.database
    async with db.sessionmaker() as s:
        g = await s.get(Generation, __import__("uuid").UUID(gid))
        g.status = GenerationStatus.PROCESSING
        await s.commit()
    assert await db.fail_interrupted_generations() == 1
    body = (await client.get(f"/generations/{gid}", headers=headers)).json()
    assert body["status"] == "failed" and "restarted" in body["error_message"]
