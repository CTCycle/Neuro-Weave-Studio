# Copyright © 2026 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator

from common.path import (
    CACHE_ROOT,
    REPOSITORY_ROOT,
    RUNTIME_DISCOVERY_CACHE_PATH,
    RUNTIME_EXPORT_ROOT,
    RUNTIME_LOG_ROOT,
    RUNTIME_ROOT,
)


###############################################################################
class Settings(BaseModel):
    """Validated application settings persisted by the selected database."""

    model_config = ConfigDict(extra="forbid")

    app_name: str = "NeuroWeave Studio"
    app_version: str = "1.0.0"
    cors_origins: str = "http://127.0.0.1:5173,http://localhost:5173"
    log_level: str = Field(default="INFO", pattern=r"^(?i:DEBUG|INFO|WARNING|ERROR|CRITICAL)$")
    log_file: Path = RUNTIME_LOG_ROOT / "backend.log"
    export_root: Path = RUNTIME_EXPORT_ROOT
    ml_job_root: Path = RUNTIME_ROOT / "ml-jobs"
    tensor_inspection_root: Path = RUNTIME_ROOT / "tensor-inspections"
    tensor_inspection_ttl_seconds: int = Field(default=600, ge=60, le=86_400)
    ml_worker_timeout_seconds: int = Field(default=300, ge=1, le=86_400)
    runtime_discovery_timeout_seconds: int = Field(default=60, ge=1, le=600)
    ml_cancel_grace_seconds: int = Field(default=5, ge=1, le=300)
    ml_terminate_grace_seconds: int = Field(default=5, ge=1, le=300)
    ml_job_history_limit: int = Field(default=100, ge=10, le=1_000)
    runtime_discovery_path: Path = RUNTIME_DISCOVERY_CACHE_PATH

    # -------------------------------------------------------------------------
    @field_validator(
        "log_file",
        "export_root",
        "ml_job_root",
        "tensor_inspection_root",
        "runtime_discovery_path",
        mode="before",
    )
    @classmethod
    def resolve_relative_paths(cls, value: object) -> object:
        if value is None:
            return value
        path = Path(str(value)).expanduser()
        return path if path.is_absolute() else (REPOSITORY_ROOT / path).resolve()

    # -------------------------------------------------------------------------
    @field_validator("runtime_discovery_path")
    @classmethod
    def validate_runtime_discovery_cache_path(cls, value: Path) -> Path:
        path = value.resolve()
        try:
            path.relative_to(CACHE_ROOT.resolve())
        except ValueError as error:
            raise ValueError(
                f"runtime_discovery_path must be beneath the canonical cache root {CACHE_ROOT}"
            ) from error
        return path

    # -------------------------------------------------------------------------
    @field_validator("cors_origins")
    @classmethod
    def validate_cors_origins(cls, value: str) -> str:
        origins = [item.strip() for item in value.split(",") if item.strip()]
        if not origins:
            raise ValueError("At least one CORS origin is required")
        return ",".join(origins)

    # -------------------------------------------------------------------------
    @property
    def cors_origin_list(self) -> list[str]:
        return [item.strip() for item in self.cors_origins.split(",") if item.strip()]


###############################################################################
class DatabaseSettings(BaseModel):
    """Validated SQLite connection parameters shared by repositories."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    sqlite_path: Path
    connect_timeout: int = Field(default=30, ge=1, le=300)


###############################################################################
def get_settings() -> Settings:
    """Return code defaults; runtime values are loaded from the active database."""

    return Settings()


###############################################################################
def get_database_settings() -> DatabaseSettings:
    """Read connection settings from the independent database configuration file."""

    from configurations.database_config import load_effective_database_config

    return load_effective_database_config().to_database_settings()
