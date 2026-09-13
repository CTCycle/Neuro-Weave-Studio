from __future__ import annotations

from pathlib import Path

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from common.path import (
    DATABASE_PATH,
    ENV_FILE,
    REPOSITORY_ROOT,
    RUNTIME_EXPORT_ROOT,
    RUNTIME_LOG_ROOT,
    RUNTIME_ROOT,
)
from configurations.environment import ensure_environment_file

###############################################################################
class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ENV_FILE,
        env_prefix="NEURO_WEAVE_",
        extra="ignore",
    )

    app_name: str = "NeuroWeave Studio"
    app_version: str = "1.0.0"
    cors_origins: str = "http://127.0.0.1:5173,http://localhost:5173"
    log_level: str = "INFO"
    log_file: Path = RUNTIME_LOG_ROOT / "backend.log"
    export_root: Path = RUNTIME_EXPORT_ROOT
    ml_job_root: Path = RUNTIME_ROOT / "ml-jobs"
    ml_worker_timeout_seconds: int = Field(default=300, ge=1, le=86_400)
    runtime_discovery_timeout_seconds: int = Field(default=60, ge=1, le=600)
    ml_cancel_grace_seconds: int = Field(default=5, ge=1, le=300)
    ml_terminate_grace_seconds: int = Field(default=5, ge=1, le=300)
    ml_job_history_limit: int = Field(default=100, ge=10, le=1_000)
    runtime_discovery_path: Path = RUNTIME_ROOT / "environment-discovery.json"

    # -------------------------------------------------------------------------
    @field_validator("export_root", mode="before")
    @classmethod
    def resolve_relative_export_root(cls, value: object) -> object:
        if value is None:
            return value
        path = Path(str(value))
        return path if path.is_absolute() else (REPOSITORY_ROOT / path).resolve()

    # -------------------------------------------------------------------------
    @field_validator("log_file", mode="before")
    @classmethod
    def resolve_relative_log_file(cls, value: object) -> object:
        if value is None:
            return value
        path = Path(str(value))
        return path if path.is_absolute() else (REPOSITORY_ROOT / path).resolve()

    # -------------------------------------------------------------------------
    @field_validator("ml_job_root", mode="before")
    @classmethod
    def resolve_relative_ml_job_root(cls, value: object) -> object:
        if value is None:
            return value
        path = Path(str(value))
        return path if path.is_absolute() else (REPOSITORY_ROOT / path).resolve()

    # -------------------------------------------------------------------------
    @field_validator("runtime_discovery_path", mode="before")
    @classmethod
    def resolve_relative_runtime_discovery_path(cls, value: object) -> object:
        if value is None:
            return value
        path = Path(str(value))
        return path if path.is_absolute() else (REPOSITORY_ROOT / path).resolve()

    # -------------------------------------------------------------------------
    @property
    def cors_origin_list(self) -> list[str]:
        return [item.strip() for item in self.cors_origins.split(",") if item.strip()]

###############################################################################
def get_settings() -> Settings:
    ensure_environment_file()
    return Settings()

###############################################################################
class _DatabaseEnvironment(BaseSettings):
    """Raw SQLite settings loaded from the process environment and `.env`."""

    model_config = SettingsConfigDict(
        env_file=ENV_FILE,
        env_prefix="",
        extra="ignore",
        case_sensitive=False,
    )

    sqlite_path: str | None = Field(
        default=None,
        validation_alias=AliasChoices("DATABASE_SQLITE_PATH", "database_sqlite_path"),
    )
    connect_timeout: int = Field(
        default=30,
        ge=1,
        validation_alias=AliasChoices("DATABASE_CONNECT_TIMEOUT", "database_connect_timeout"),
    )

###############################################################################
class DatabaseSettings(BaseModel):
    """Validated, immutable SQLite configuration shared by repositories."""

    model_config = ConfigDict(frozen=True)

    sqlite_path: Path
    connect_timeout: int = Field(default=30, ge=1)

###############################################################################
def _resolve_path(value: str | None, default: Path) -> Path:
    path = Path(value.strip()) if value and value.strip() else default
    return path if path.is_absolute() else (REPOSITORY_ROOT / path).resolve()

###############################################################################
def get_database_settings() -> DatabaseSettings:
    ensure_environment_file()
    environment = _DatabaseEnvironment()
    return DatabaseSettings(
        sqlite_path=_resolve_path(
            environment.sqlite_path,
            DATABASE_PATH,
        ),
        connect_timeout=environment.connect_timeout,
    )
