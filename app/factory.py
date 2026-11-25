"""Application factory. Each dependency can be injected, so tests use an in-memory-style setup."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.config import Settings, get_settings
from app.db import Database
from app.routers import auth, generations
from app.services.generations import GenerationService
from app.services.inference_client import InferenceClient
from app.services.storage import LocalStorage


def create_app(settings: Settings | None = None, inference: InferenceClient | None = None) -> FastAPI:
    settings = settings or get_settings()
    database = Database(settings.database_url)
    storage = LocalStorage(settings.storage_dir)
    inference = inference or InferenceClient(
        base_url=settings.inference_base_url,
        api_key=settings.inference_api_key,
        timeout_seconds=settings.inference_timeout_seconds,
    )
    generation_service = GenerationService(
        database.sessionmaker,
        inference,
        storage,
        max_concurrent=settings.max_concurrent_generations,
    )

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        if settings.create_tables_on_startup:
            await database.create_all()
        await database.fail_interrupted_generations()
        yield
        await database.dispose()

    app = FastAPI(title="Renderforge API", version="0.2.0", lifespan=lifespan)
    app.state.settings = settings
    app.state.database = database
    app.state.storage = storage
    app.state.generations = generation_service

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(auth.router)
    app.include_router(generations.router)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/health/db")
    async def health_db() -> dict[str, str]:
        try:
            await database.check()
        except Exception as exc:  # any driver error means "not ready"
            raise HTTPException(503, "Database unreachable") from exc
        return {"status": "ok"}

    return app
