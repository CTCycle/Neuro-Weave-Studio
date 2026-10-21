# Copyright © 2026 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, ValidationError

from api.dependencies import get_repository_context
from common.logging_config import configure_logging
from configurations.database_config import DatabaseConfig
from configurations.settings import Settings
from services.database_configuration_service import (
    DatabaseConfigurationService,
    DatabaseSwitchBusyError,
)
from services.runtime_controller import RuntimeController, RuntimeControllerError

router = APIRouter(prefix="/api/settings", tags=["settings"])
LOGGER = logging.getLogger(__name__)


###############################################################################
class DatabaseSettingsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    backend: str = "sqlite"
    sqlite_path: str
    connect_timeout_seconds: int = 30
    schema_version: int = 1

    # -------------------------------------------------------------------------
    def to_config(self) -> DatabaseConfig:
        return DatabaseConfig.model_validate(self.model_dump())


###############################################################################
class DatabaseStatusResponse(BaseModel):
    backend: str
    sqlite_path: str | None
    connect_timeout_seconds: int | None
    available: bool
    switching: bool
    message: str | None


###############################################################################
class DatabaseTestResponse(BaseModel):
    success: bool
    backend: str
    sqlite_path: str
    message: str


###############################################################################
class DatabaseSwitchResponse(BaseModel):
    status: DatabaseStatusResponse
    settings: Settings


###############################################################################
def _status_response(database_configuration: DatabaseConfigurationService) -> DatabaseStatusResponse:
    status = database_configuration.status()
    return DatabaseStatusResponse(
        backend=status.backend,
        sqlite_path=status.sqlite_path,
        connect_timeout_seconds=status.connect_timeout_seconds,
        available=status.available,
        switching=status.switching,
        message=status.message,
    )


###############################################################################
@router.get("", response_model=Settings)
def read_settings(request: Request) -> Settings:
    context = get_repository_context(request)
    with context.new_unit_of_work() as uow:
        return uow.application_settings.get() or request.app.state.settings


@router.patch("", response_model=Settings)
def patch_settings(
    payload: dict[str, object],
    request: Request,
) -> Settings:
    context = get_repository_context(request)
    with context.new_unit_of_work() as uow:
        previous = uow.application_settings.get() or request.app.state.settings
        patch = dict(payload)
        preferences = patch.get("runtime_preferences")
        if preferences is not None:
            if not isinstance(preferences, dict):
                raise HTTPException(status_code=422, detail="runtime_preferences must be an object.")
            patch["runtime_preferences"] = {
                **previous.runtime_preferences,
                **preferences,
            }
        try:
            updated = Settings.model_validate({**previous.model_dump(), **patch})
        except ValidationError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
    return _save_application_settings(updated, request)


###############################################################################
def _save_application_settings(payload: Settings, request: Request) -> Settings:
    context = get_repository_context(request)
    controller: RuntimeController = request.app.state.runtime_controller
    with context.new_unit_of_work() as uow:
        previous = uow.application_settings.get() or request.app.state.settings
        try:
            controller.apply_settings(payload)
            configure_logging(payload.log_level, payload.log_file)
            uow.application_settings.save(payload)
            uow.commit()
        except RuntimeControllerError as error:
            controller.apply_settings(previous)
            configure_logging(previous.log_level, previous.log_file)
            raise HTTPException(status_code=409, detail=str(error)) from error
        except Exception as error:
            controller.apply_settings(previous)
            configure_logging(previous.log_level, previous.log_file)
            LOGGER.exception("Could not persist application settings")
            raise HTTPException(status_code=500, detail="Application settings could not be saved.") from error
    request.app.state.settings = payload
    return payload


###############################################################################
@router.get("/database", response_model=DatabaseStatusResponse)
def read_database_status(request: Request) -> DatabaseStatusResponse:
    database_configuration: DatabaseConfigurationService = request.app.state.database_configuration
    return _status_response(database_configuration)


###############################################################################
@router.post("/database/test", response_model=DatabaseTestResponse)
async def test_database_connection(
    payload: DatabaseSettingsRequest,
    request: Request,
) -> DatabaseTestResponse:
    database_configuration: DatabaseConfigurationService = request.app.state.database_configuration
    try:
        candidate = payload.to_config()
        await database_configuration.test(candidate)
    except Exception as error:
        raise HTTPException(status_code=422, detail=f"Connection test failed: {error}") from error
    return DatabaseTestResponse(
        success=True,
        backend=candidate.backend,
        sqlite_path=str(candidate.sqlite_path),
        message="SQLite connection and required schema are ready.",
    )


###############################################################################
@router.put("/database", response_model=DatabaseSwitchResponse)
async def switch_database(payload: DatabaseSettingsRequest, request: Request) -> DatabaseSwitchResponse:
    database_configuration: DatabaseConfigurationService = request.app.state.database_configuration
    controller: RuntimeController = request.app.state.runtime_controller
    previous_settings: Settings = request.app.state.settings

    def apply_settings(settings: Settings) -> None:
        controller.apply_settings(settings)
        request.app.state.settings = settings
        configure_logging(settings.log_level, settings.log_file)

    try:
        candidate = payload.to_config()
        settings = await database_configuration.activate(
            candidate,
            has_active_jobs=lambda: bool(controller.list(active_only=True)),
            apply_settings=apply_settings,
        )
    except DatabaseSwitchBusyError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except RuntimeControllerError as error:
        apply_settings(previous_settings)
        raise HTTPException(status_code=409, detail=str(error)) from error
    except Exception as error:
        apply_settings(previous_settings)
        LOGGER.exception("Database activation failed")
        raise HTTPException(status_code=422, detail=f"Database was not switched: {error}") from error

    request.app.state.repository_context = database_configuration.repository_context
    return DatabaseSwitchResponse(
        status=_status_response(database_configuration),
        settings=settings,
    )
