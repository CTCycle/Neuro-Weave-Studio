from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from neuro_weave.paths import (
    REPOSITORY_ROOT,
    RUNTIME_EXPORT_ROOT,
    RUNTIME_LOG_ROOT,
    RUNTIME_ROOT,
    SETTINGS_ROOT,
)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=SETTINGS_ROOT / ".env",
        env_prefix="NEURO_WEAVE_",
        extra="ignore",
    )

    app_name: str = "NeuroWeave Studio"
    app_version: str = "0.1.0"
    database_url: str = f"sqlite:///{(RUNTIME_ROOT / 'neuro_weave.db').as_posix()}"
    cors_origins: str = "http://127.0.0.1:5173,http://localhost:5173"
    log_level: str = "INFO"
    log_file: Path = RUNTIME_LOG_ROOT / "backend.log"
    export_root: Path = RUNTIME_EXPORT_ROOT

    @field_validator("database_url", mode="before")
    @classmethod
    def resolve_relative_sqlite_url(cls, value: object) -> object:
        if not isinstance(value, str):
            return value
        prefix = "sqlite:///"
        if not value.startswith(prefix):
            return value
        raw_path = value.removeprefix(prefix)
        path = Path(raw_path)
        if path.is_absolute():
            return value
        normalized = raw_path.removeprefix("./")
        return f"{prefix}{(REPOSITORY_ROOT / normalized).resolve().as_posix()}"

    @field_validator("export_root", mode="before")
    @classmethod
    def resolve_relative_export_root(cls, value: object) -> object:
        if value is None:
            return value
        path = Path(str(value))
        return path if path.is_absolute() else (REPOSITORY_ROOT / path).resolve()

    @field_validator("log_file", mode="before")
    @classmethod
    def resolve_relative_log_file(cls, value: object) -> object:
        if value is None:
            return value
        path = Path(str(value))
        return path if path.is_absolute() else (REPOSITORY_ROOT / path).resolve()

    @property
    def cors_origin_list(self) -> list[str]:
        return [item.strip() for item in self.cors_origins.split(",") if item.strip()]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
