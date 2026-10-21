# Copyright © 2026 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from common.path import DATABASE_PATH, REPOSITORY_ROOT, RESOURCE_ROOT
from configurations.settings import DatabaseSettings

DATABASE_CONFIG_PATH = RESOURCE_ROOT / "database-config.json"
CONFIG_SCHEMA_VERSION = 1

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
def bootstrap_database_config(config_path: Path = DATABASE_CONFIG_PATH) -> tuple[DatabaseConfig, bool]:
    """Load the config file, or prepare a repository-default config when it is missing.

    Returns the configuration and whether the file had to be seeded.
    """

    target = Path(config_path)
    if target.exists():
        payload = json.loads(target.read_text(encoding="utf-8"))
        return DatabaseConfig.model_validate(payload), False
    return DatabaseConfig(), True

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
