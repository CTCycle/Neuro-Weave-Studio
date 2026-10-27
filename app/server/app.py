# Copyright © 2026 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path, PurePosixPath

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from api.routes import (
    catalog,
    compilation,
    datasets,
    execution,
    exports,
    filesystem,
    health,
    imports,
    metadata,
    metrics,
    projects,
    tensor_inspections,
    validation,
)
from api.routes import (
    settings as settings_routes,
)
from common import path as common_path
from common.catalogs.loader import get_node_catalog
from common.logging_config import configure_logging
from configurations.database_config import DATABASE_CONFIG_PATH
from configurations.runtime_config import RuntimeConfiguration, load_runtime_configuration
from configurations.settings import Settings, get_settings
from services.database_configuration_service import DatabaseConfigurationService
from services.runtime_controller import RuntimeController

LOGGER = logging.getLogger(__name__)

###############################################################################
def _log_startup_stage(stage: str, started: float) -> None:
    LOGGER.debug(
        "startup_stage stage=%s duration_ms=%s",
        stage,
        round((time.perf_counter() - started) * 1000, 1),
    )

###############################################################################
@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncGenerator[None]:
    startup_started = time.perf_counter()
    database_configuration: DatabaseConfigurationService = application.state.database_configuration
    stage_started = time.perf_counter()
    await asyncio.to_thread(database_configuration.start)
    settings = database_configuration.settings
    application.state.settings = settings
    application.state.repository_context = database_configuration.repository_context
    configure_logging(settings.log_level, settings.log_file)
    _log_startup_stage("database_configuration", stage_started)

    stage_started = time.perf_counter()
    settings.export_root.mkdir(parents=True, exist_ok=True)
    _log_startup_stage("export_root", stage_started)

    stage_started = time.perf_counter()
    catalog = get_node_catalog()
    _log_startup_stage("node_catalog", stage_started)

    stage_started = time.perf_counter()
    runtime_controller = RuntimeController(settings)
    _log_startup_stage("runtime_controller", stage_started)
    application.state.catalog = catalog
    application.state.runtime_controller = runtime_controller
    LOGGER.info("Application started; runtime discovery is on demand")
    _log_startup_stage("lifespan", startup_started)
    try:
        yield
    finally:
        runtime_controller.shutdown()
        database_configuration.dispose()
        LOGGER.info("Application shutdown complete")

###############################################################################
def _client_build_available() -> bool:
    return (common_path.FRONTEND_DIST_ROOT / "index.html").is_file()

###############################################################################
def _resolve_client_file(full_path: str) -> Path | None:
    client_root = common_path.FRONTEND_DIST_ROOT.resolve()
    requested_path = (client_root / full_path).resolve()
    if not requested_path.is_relative_to(client_root):
        return None
    return requested_path if requested_path.is_file() else None

###############################################################################
def _serve_client_root() -> FileResponse:
    return FileResponse(common_path.FRONTEND_DIST_ROOT / "index.html")

###############################################################################
def _serve_client_path(full_path: str) -> FileResponse:
    client_file = _resolve_client_file(full_path)
    if client_file is not None:
        return FileResponse(client_file)
    if PurePosixPath(full_path).suffix:
        raise HTTPException(status_code=404, detail="Static file not found")
    return _serve_client_root()

###############################################################################
def _raise_unknown_api_root() -> None:
    raise HTTPException(status_code=404, detail="API route not found")

###############################################################################
def _raise_unknown_api_route(_full_path: str) -> None:
    raise HTTPException(status_code=404, detail="API route not found")

###############################################################################
def _redirect_root_to_docs() -> RedirectResponse:
    return RedirectResponse("/docs")

###############################################################################
def _serve_runtime_configuration(request: Request) -> JSONResponse:
    runtime_configuration: RuntimeConfiguration = request.app.state.runtime_configuration
    return JSONResponse(
        {"api_base_url": runtime_configuration.api_base_url},
        headers={"Cache-Control": "no-store"},
    )

###############################################################################
def create_app(
    *,
    database_config_path: Path = DATABASE_CONFIG_PATH,
    initial_settings: Settings | None = None,
) -> FastAPI:
    settings = get_settings()
    runtime_configuration = load_runtime_configuration()
    application = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="Framework-independent visual model architecture service",
        lifespan=lifespan,
    )
    application.state.settings = settings
    application.state.runtime_configuration = runtime_configuration
    application.state.database_configuration = DatabaseConfigurationService(
        database_config_path,
        initial_settings=initial_settings,
    )
    application.state.repository_context = None
    # SQLite readers can run concurrently, but shared-process mutations must
    # not attempt to upgrade separate deferred transactions at the same time.
    application.state.sqlite_write_lock = asyncio.Lock()
    async def log_api_request(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        origin = request.headers.get("origin")
        runtime_configuration: RuntimeConfiguration = request.app.state.runtime_configuration
        allowed_origin = origin in runtime_configuration.cors_origin_list if origin else False
        preflight = (
            request.method == "OPTIONS"
            and request.headers.get("access-control-request-method") is not None
        )
        if preflight:
            response = Response(status_code=200 if allowed_origin else 400)
            LOGGER.log(
                logging.DEBUG if allowed_origin else logging.WARNING,
                "api_preflight method=%s path=%s status=%s",
                request.method,
                request.url.path,
                response.status_code,
            )
        elif not request.url.path.startswith("/api"):
            return await call_next(request)
        else:
            started = time.perf_counter()
            database_configuration: DatabaseConfigurationService = request.app.state.database_configuration
            is_database_configuration_route = request.url.path.startswith("/api/settings/database")
            request_entered = False
            try:
                if not is_database_configuration_route:
                    await database_configuration.enter_request()
                    request_entered = True
                    if not database_configuration.available:
                        response = JSONResponse(
                            {"detail": "The configured database is unavailable. Check Database settings and try again."},
                            status_code=503,
                        )
                    elif request.method in {"POST", "PUT", "PATCH", "DELETE"}:
                        async with request.app.state.sqlite_write_lock:
                            response = await call_next(request)
                    else:
                        response = await call_next(request)
                else:
                    response = await call_next(request)
            except Exception:
                LOGGER.exception(
                    "api_request method=%s path=%s status=exception duration_ms=%s",
                    request.method,
                    request.url.path,
                    round((time.perf_counter() - started) * 1000, 1),
                )
                raise
            finally:
                if request_entered:
                    await database_configuration.leave_request()
            log_level = logging.ERROR if response.status_code >= 500 else (
                logging.WARNING if response.status_code >= 400 else logging.DEBUG
            )
            LOGGER.log(
                log_level,
                "api_request method=%s path=%s status=%s duration_ms=%s",
                request.method,
                request.url.path,
                response.status_code,
                round((time.perf_counter() - started) * 1000, 1),
            )
        if allowed_origin and origin is not None:
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, PATCH, DELETE, OPTIONS"
            requested_headers = request.headers.get("access-control-request-headers")
            response.headers["Access-Control-Allow-Headers"] = requested_headers or "*"
            response.headers["Vary"] = "Origin"
        return response

    application.middleware("http")(log_api_request)

    application.include_router(health.router)
    application.include_router(metadata.router)
    application.include_router(metrics.router)
    application.include_router(catalog.router)
    application.include_router(projects.router)
    application.include_router(datasets.router)
    application.include_router(validation.router)
    application.include_router(tensor_inspections.router)
    application.include_router(compilation.router)
    application.include_router(execution.router)
    application.include_router(exports.router)
    application.include_router(imports.router)
    application.include_router(filesystem.router)
    application.include_router(settings_routes.router)
    application.add_api_route(
        "/api",
        _raise_unknown_api_root,
        methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"],
        include_in_schema=False,
    )
    application.add_api_route(
        "/api/{_full_path:path}",
        _raise_unknown_api_route,
        methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"],
        include_in_schema=False,
    )
    application.add_api_route(
        "/runtime-config.json",
        _serve_runtime_configuration,
        methods=["GET"],
        include_in_schema=False,
    )

    if _client_build_available():
        if common_path.FRONTEND_ASSETS_ROOT.is_dir():
            application.mount(
                "/assets",
                StaticFiles(directory=str(common_path.FRONTEND_ASSETS_ROOT)),
                name="neuroweave-assets",
            )
        application.add_api_route(
            "/",
            _serve_client_root,
            methods=["GET"],
            include_in_schema=False,
        )
        application.add_api_route(
            "/{full_path:path}",
            _serve_client_path,
            methods=["GET"],
            include_in_schema=False,
        )
    else:
        application.add_api_route(
            "/",
            _redirect_root_to_docs,
            methods=["GET"],
            include_in_schema=False,
        )
    return application
