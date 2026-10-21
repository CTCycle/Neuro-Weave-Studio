# Copyright © 2026 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from urllib.parse import SplitResult, urlsplit

from pydantic import BaseModel, ConfigDict, field_validator

from common.path import ENV_FILE

DEFAULT_API_BASE_URL = "/api"
DEFAULT_CORS_ORIGINS = "http://127.0.0.1:5173,http://localhost:5173"
API_BASE_URL_ENV = "NEURO_WEAVE_API_BASE_URL"
CORS_ORIGINS_ENV = "NEURO_WEAVE_CORS_ORIGINS"


def _has_valid_port(parsed: SplitResult) -> bool:
    try:
        port = parsed.port
    except ValueError:
        return False
    return port is None or 1 <= port <= 65_535


class RuntimeConfiguration(BaseModel):
    """Process-level deployment settings, loaded independently of the database."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    api_base_url: str = DEFAULT_API_BASE_URL
    cors_origins: str = DEFAULT_CORS_ORIGINS

    @field_validator("api_base_url", mode="before")
    @classmethod
    def validate_api_base_url(cls, value: object) -> str:
        candidate = str(value).strip()
        if not candidate:
            return DEFAULT_API_BASE_URL
        if any(character.isspace() for character in candidate):
            raise ValueError("API base URL cannot contain whitespace")

        parsed = urlsplit(candidate)
        if candidate.startswith("/") and not candidate.startswith("//"):
            if "?" in candidate or "#" in candidate or parsed.path != candidate:
                raise ValueError("Relative API base URL must be a path without query or fragment")
            return parsed.path.rstrip("/") or "/"

        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("API base URL must be a relative path or an absolute HTTP(S) URL")
        if (
            parsed.username is not None
            or parsed.password is not None
            or "?" in candidate
            or "#" in candidate
        ):
            raise ValueError("API base URL cannot contain credentials, query, or fragment")
        if not _has_valid_port(parsed):
            raise ValueError("API base URL contains an invalid port")
        return candidate.rstrip("/")

    @field_validator("cors_origins", mode="before")
    @classmethod
    def validate_cors_origins(cls, value: object) -> str:
        entries = [entry.strip() for entry in str(value).split(",") if entry.strip()]
        if not entries:
            raise ValueError("At least one CORS origin is required")

        origins: list[str] = []
        for entry in entries:
            parsed = urlsplit(entry)
            if (
                any(character.isspace() for character in entry)
                or parsed.scheme not in {"http", "https"}
                or not parsed.netloc
                or parsed.path not in {"", "/"}
                or parsed.query
                or parsed.fragment
                or "?" in entry
                or "#" in entry
                or parsed.username is not None
                or parsed.password is not None
                or not _has_valid_port(parsed)
            ):
                raise ValueError(f"Invalid CORS origin: {entry}")
            origins.append(f"{parsed.scheme.lower()}://{parsed.netloc.lower()}")

        return ",".join(dict.fromkeys(origins))

    @property
    def cors_origin_list(self) -> list[str]:
        return self.cors_origins.split(",")


def load_runtime_configuration(
    environ: Mapping[str, str] | None = None,
    env_file: Path = ENV_FILE,
) -> RuntimeConfiguration:
    """Load deployment settings, preferring inherited process variables over .env."""

    values = _read_environment_file(Path(env_file))
    process_values = os.environ if environ is None else environ
    values.update(
        {
            key: value
            for key, value in process_values.items()
            if key in {API_BASE_URL_ENV, CORS_ORIGINS_ENV}
        }
    )
    return RuntimeConfiguration(
        api_base_url=values.get(API_BASE_URL_ENV, DEFAULT_API_BASE_URL),
        cors_origins=values.get(CORS_ORIGINS_ENV, DEFAULT_CORS_ORIGINS),
    )


def _read_environment_file(path: Path) -> dict[str, str]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return {}

    values: dict[str, str] = {}
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith(("#", ";")):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        name, separator, raw_value = line.partition("=")
        if not separator or not name.strip():
            continue
        value = raw_value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        else:
            value = value.split("#", 1)[0].rstrip()
        values[name.strip()] = value
    return values
