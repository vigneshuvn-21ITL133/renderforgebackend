"""Runtime configuration, loaded from environment variables prefixed with RF_."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="RF_", extra="ignore")

    database_url: str = "postgresql+asyncpg://renderforge:renderforge@localhost:5432/renderforge"
    create_tables_on_startup: bool = Field(
        default=True,
        description="Create tables from the models at startup. Replace with Alembic migrations before production.",
    )

    db_pool_size: int = Field(default=5, ge=1)
    db_max_overflow: int = Field(default=5, ge=0)
    db_pool_recycle_seconds: int = Field(default=240, ge=30, description="Keep below Neon's idle-connection timeout")

    jwt_secret_key: str = Field(min_length=32)
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = Field(default=30, ge=1)
    refresh_token_expire_days: int = Field(default=14, ge=1)

    allowed_origins: list[str] = ["http://localhost:3000"]

    inference_base_url: str = "http://localhost:8001"
    inference_api_key: str = Field(min_length=16)
    inference_timeout_seconds: float = Field(default=600.0, gt=0)
    max_concurrent_generations: int = Field(default=1, ge=1, description="Match the number of GPUs")

    storage_dir: Path = Path("storage")
    max_upload_mb: int = Field(default=10, ge=1)

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    return Settings()
