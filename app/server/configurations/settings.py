from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
APP_ROOT = REPOSITORY_ROOT / "app"
SETTINGS_ROOT = REPOSITORY_ROOT / "settings"
RUNTIME_ROOT = APP_ROOT / "resources" / "runtime"
RUNTIME_LOG_ROOT = RUNTIME_ROOT / "logs"
RUNTIME_EXPORT_ROOT = RUNTIME_ROOT / "exports"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=SETTINGS_ROOT / ".env",
        env_prefix="NEURO_WEAVE_",
        extra="ignore",
    )

    app_name: str = "NeuroWeave Studio"
    app_version: str = "0.1.0"
    cors_origins: str = "http://127.0.0.1:5173,http://localhost:5173"
    log_level: str = "INFO"
    log_file: Path = RUNTIME_LOG_ROOT / "backend.log"
    export_root: Path = RUNTIME_EXPORT_ROOT

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


class _DatabaseEnvironment(BaseSettings):
    """Raw database environment values, including the local `.env` file."""

    model_config = SettingsConfigDict(
        env_file=SETTINGS_ROOT / ".env",
        env_prefix="",
        extra="ignore",
        case_sensitive=False,
    )

    embedded_database: str | None = Field(
        default=None,
        validation_alias=AliasChoices("DATABASE_EMBEDDED", "database_embedded"),
    )
    url: str | None = Field(
        default=None,
        validation_alias=AliasChoices("DATABASE_URL", "database_url"),
    )
    engine: str | None = Field(
        default=None,
        validation_alias=AliasChoices("DATABASE_ENGINE", "database_engine"),
    )
    host: str | None = Field(
        default=None,
        validation_alias=AliasChoices("DATABASE_HOST", "database_host"),
    )
    port: str | None = Field(
        default=None,
        validation_alias=AliasChoices("DATABASE_PORT", "database_port"),
    )
    name: str | None = Field(
        default=None,
        validation_alias=AliasChoices("DATABASE_NAME", "database_name"),
    )
    username: str | None = Field(
        default=None,
        validation_alias=AliasChoices("DATABASE_USERNAME", "database_username"),
    )
    password: str | None = Field(
        default=None,
        validation_alias=AliasChoices("DATABASE_PASSWORD", "database_password"),
    )
    ssl: str | None = Field(
        default=None,
        validation_alias=AliasChoices("DATABASE_SSL", "database_ssl"),
    )
    ssl_ca: str | None = Field(
        default=None,
        validation_alias=AliasChoices("DATABASE_SSL_CA", "database_ssl_ca"),
    )
    connect_timeout: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "DATABASE_CONNECT_TIMEOUT", "database_connect_timeout"
        ),
    )
    insert_batch_size: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "DATABASE_INSERT_BATCH_SIZE", "database_insert_batch_size"
        ),
    )
    sqlite_path: str | None = Field(
        default=None,
        validation_alias=AliasChoices("DATABASE_SQLITE_PATH", "database_sqlite_path"),
    )


class DatabaseSettings(BaseModel):
    """Validated, immutable database configuration shared by repositories."""

    model_config = ConfigDict(frozen=True)

    embedded_database: bool
    url: str | None = None
    sqlite_path: Path | None = None
    engine: str | None = None
    host: str | None = None
    port: int | None = None
    database_name: str | None = None
    username: str | None = None
    password: str | None = None
    ssl: bool = False
    ssl_ca: str | None = None
    connect_timeout: int = Field(default=30, ge=1)
    insert_batch_size: int = Field(default=1000, ge=1)


_TRUE_VALUES = {"1", "true", "yes", "on"}
_FALSE_VALUES = {"0", "false", "no", "off"}


def _optional_text(value: object | None) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _strict_bool(value: str | None, *, default: bool) -> bool:
    normalized = _optional_text(value)
    if normalized is None:
        return default
    lowered = normalized.casefold()
    if lowered in _TRUE_VALUES:
        return True
    if lowered in _FALSE_VALUES:
        return False
    raise ValueError("DATABASE_EMBEDDED and DATABASE_SSL must be boolean values")


def _positive_int(value: str | None, *, default: int, label: str) -> int:
    normalized = _optional_text(value)
    if normalized is None:
        return default
    try:
        parsed = int(normalized)
    except ValueError as exc:
        raise ValueError(f"{label} must be an integer") from exc
    if parsed < 1:
        raise ValueError(f"{label} must be greater than zero")
    return parsed


def _url_database_values(url: str | None) -> dict[str, Any]:
    if not url:
        return {}
    parsed = urlparse(url)
    if parsed.scheme in {"sqlite", "sqlite+pysqlite"}:
        return {}
    query = parse_qs(parsed.query, keep_blank_values=False)
    return {
        "engine": parsed.scheme or None,
        "host": parsed.hostname,
        "port": parsed.port,
        "database_name": unquote(parsed.path.lstrip("/")) or None,
        "username": unquote(parsed.username) if parsed.username else None,
        "password": unquote(parsed.password) if parsed.password else None,
        "ssl": (query.get("sslmode", [""])[0].casefold() not in {"", "disable", "false", "off", "0"})
        if "sslmode" in query
        else None,
        "connect_timeout": query.get("connect_timeout", [None])[0],
    }


def _resolve_path(value: str | None, default: Path) -> Path:
    raw = _optional_text(value)
    path = Path(raw) if raw else default
    return path if path.is_absolute() else (REPOSITORY_ROOT / path).resolve()


@lru_cache(maxsize=1)
def get_database_settings() -> DatabaseSettings:
    environment = _DatabaseEnvironment()
    embedded = _strict_bool(environment.embedded_database, default=True)
    explicit_ssl = _strict_bool(environment.ssl, default=False)
    insert_batch_size = _positive_int(
        environment.insert_batch_size,
        default=1000,
        label="DATABASE_INSERT_BATCH_SIZE",
    )

    if embedded:
        return DatabaseSettings(
            embedded_database=True,
            sqlite_path=_resolve_path(
                environment.sqlite_path,
                RUNTIME_ROOT / "neuro_weave.db",
            ),
            connect_timeout=_positive_int(
                environment.connect_timeout,
                default=30,
                label="DATABASE_CONNECT_TIMEOUT",
            ),
            insert_batch_size=insert_batch_size,
        )

    url_values = _url_database_values(_optional_text(environment.url))

    def explicit_or_url(name: str, value: str | None) -> Any:
        explicit = _optional_text(value)
        return explicit if explicit is not None else url_values.get(name)

    engine = explicit_or_url("engine", environment.engine) or "postgresql+psycopg"
    host = explicit_or_url("host", environment.host)
    database_name = explicit_or_url("database_name", environment.name)
    username = explicit_or_url("username", environment.username)
    password = explicit_or_url("password", environment.password)
    if not host or not database_name or not username:
        raise ValueError(
            "External database mode requires DATABASE_HOST, DATABASE_NAME, and DATABASE_USERNAME"
        )
    port_value = explicit_or_url("port", environment.port)
    port = _positive_int(port_value, default=5432, label="DATABASE_PORT")
    if port > 65535:
        raise ValueError("DATABASE_PORT must be between 1 and 65535")
    url_timeout = url_values.get("connect_timeout")
    connect_timeout = _positive_int(
        environment.connect_timeout or url_timeout,
        default=30,
        label="DATABASE_CONNECT_TIMEOUT",
    )
    ssl = (
        explicit_ssl
        if environment.ssl is not None
        else bool(url_values.get("ssl") or False)
    )
    return DatabaseSettings(
        embedded_database=False,
        url=_optional_text(environment.url),
        engine=str(engine).casefold(),
        host=str(host),
        port=port,
        database_name=str(database_name),
        username=str(username),
        password=_optional_text(str(password)) if password is not None else None,
        ssl=ssl,
        ssl_ca=_optional_text(environment.ssl_ca),
        connect_timeout=connect_timeout,
        insert_batch_size=insert_batch_size,
    )
