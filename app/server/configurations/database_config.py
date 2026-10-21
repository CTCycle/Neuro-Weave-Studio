# Copyright © 2026 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from common.path import DATABASE_PATH, ENV_FILE, REPOSITORY_ROOT, RESOURCE_ROOT
from configurations.settings import DatabaseSettings, Settings

DATABASE_CONFIG_PATH = RESOURCE_ROOT / "database-config.json"
CONFIG_SCHEMA_VERSION = 1
DATABASE_ENV_KEYS = {"DATABASE_SQLITE_PATH", "DATABASE_CONNECT_TIMEOUT"}
APPLICATION_ENV_KEYS = {
    f"NEURO_WEAVE_{field.upper()}"
    for field in Settings.model_fields
    if field not in {
        "app_name",
        "app_version",
        "runtime_preferences",
        "runtime_preference_backend",
    }
} | {"NEURO_WEAVE_APP_NAME", "NEURO_WEAVE_APP_VERSION"}
LEGACY_ENV_KEYS = DATABASE_ENV_KEYS | APPLICATION_ENV_KEYS


###############################################################################
class DatabaseConfig(BaseModel):
    """The database-only file contract. No application settings or credentials belong here."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = CONFIG_SCHEMA_VERSION
    backend: Literal["sqlite"] = "sqlite"
    sqlite_path: Path = DATABASE_PATH
    connect_timeout_seconds: int = Field(default=30, ge=1, le=300)

    # -------------------------------------------------------------------------
    @field_validator("sqlite_path", mode="before")
    @classmethod
    def resolve_sqlite_path(cls, value: object) -> Path:
        path = Path(str(value)).expanduser()
        return path.resolve() if path.is_absolute() else (REPOSITORY_ROOT / path).resolve()

    # -------------------------------------------------------------------------
    def to_database_settings(self) -> DatabaseSettings:
        return DatabaseSettings(
            sqlite_path=self.sqlite_path,
            connect_timeout=self.connect_timeout_seconds,
        )


###############################################################################
@dataclass(frozen=True, slots=True)
class DatabaseBootstrap:
    config: DatabaseConfig
    legacy_settings: Settings | None
    config_was_missing: bool


###############################################################################
def _legacy_environment_values(environment_file: Path = ENV_FILE) -> dict[str, str]:
    values: dict[str, str] = {}
    for key in LEGACY_ENV_KEYS:
        inherited = os.environ.get(key)
        if inherited is not None and inherited.strip():
            values[key] = inherited.strip().strip('"').strip("'")

    try:
        lines = environment_file.read_text(encoding="utf-8").splitlines()
    except OSError:
        return values
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith(("#", ";")) or "=" not in line:
            continue
        name, raw_value = line.split("=", 1)
        name = name.strip().upper()
        if name in LEGACY_ENV_KEYS and name not in values:
            values[name] = raw_value.strip().strip('"').strip("'")
    return values


###############################################################################
def bootstrap_database_config(
    config_path: Path = DATABASE_CONFIG_PATH,
    environment_file: Path = ENV_FILE,
) -> DatabaseBootstrap:
    """Load the config, or prepare a one-time import without writing it yet."""

    target = Path(config_path)
    if target.exists():
        payload = json.loads(target.read_text(encoding="utf-8"))
        return DatabaseBootstrap(DatabaseConfig.model_validate(payload), None, False)

    legacy = _legacy_environment_values(environment_file)
    sqlite_path = legacy.get("DATABASE_SQLITE_PATH")
    timeout = legacy.get("DATABASE_CONNECT_TIMEOUT", "30")
    config = DatabaseConfig(
        sqlite_path=Path(sqlite_path) if sqlite_path else DATABASE_PATH,
        connect_timeout_seconds=int(timeout),
    )
    legacy_settings_payload = {
        field: legacy[f"NEURO_WEAVE_{field.upper()}"]
        for field in Settings.model_fields
        if f"NEURO_WEAVE_{field.upper()}" in legacy
    }
    legacy_settings = Settings.model_validate(legacy_settings_payload) if legacy_settings_payload else None
    return DatabaseBootstrap(config, legacy_settings, True)


###############################################################################
def load_effective_database_config(config_path: Path = DATABASE_CONFIG_PATH) -> DatabaseConfig:
    target = Path(config_path)
    if not target.is_file():
        raise FileNotFoundError(
            f"Database configuration is missing: {target}. Start the application or run app/scripts/initialize_database.py."
        )
    payload = json.loads(target.read_text(encoding="utf-8"))
    return DatabaseConfig.model_validate(payload)


###############################################################################
def write_database_config(
    config: DatabaseConfig,
    config_path: Path = DATABASE_CONFIG_PATH,
) -> None:
    target = Path(config_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(config.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(content)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, target)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


###############################################################################
def remove_legacy_environment_values(environment_file: Path = ENV_FILE) -> None:
    """Remove migrated application/database keys without disturbing other local values."""

    target = Path(environment_file)
    if not target.is_file():
        return
    original = target.read_text(encoding="utf-8")
    lines = original.splitlines(keepends=True)
    retained: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith(("#", ";")) or "=" not in stripped:
            retained.append(line)
            continue
        name = stripped.split("=", 1)[0].strip().upper()
        if name not in LEGACY_ENV_KEYS:
            retained.append(line)
    updated = "".join(retained)
    if updated == original:
        return

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(updated)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, target)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


###############################################################################
def clear_legacy_process_environment() -> None:
    for key in LEGACY_ENV_KEYS:
        os.environ.pop(key, None)
