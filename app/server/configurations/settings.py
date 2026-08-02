from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=REPOSITORY_ROOT / ".env",
        env_prefix="NEURO_WEAVE_",
        extra="ignore",
    )

    app_name: str = "NeuroWeave Studio"
    app_version: str = "0.1.0"
    database_url: str = (
        f"sqlite:///{(REPOSITORY_ROOT / 'runtime' / 'neuro_weave.db').as_posix()}"
    )
    cors_origins: str = "http://localhost:5173"
    log_level: str = "INFO"
    export_root: Path = REPOSITORY_ROOT / "runtime" / "exports"

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

    @property
    def cors_origin_list(self) -> list[str]:
        return [item.strip() for item in self.cors_origins.split(",") if item.strip()]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
