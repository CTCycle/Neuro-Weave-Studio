from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from api.routes import catalog, compilation, exports, health, metadata, projects, validation
from common import path as common_path
from common.catalogs.loader import get_node_catalog
from common.logging_config import configure_logging
from configurations.settings import get_settings
from repositories.session import close_repository_context, get_repository_context


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    configure_logging(settings.log_level)
    settings.export_root.mkdir(parents=True, exist_ok=True)
    get_node_catalog()
    get_repository_context()
    try:
        yield
    finally:
        close_repository_context()


def _client_build_available() -> bool:
    return (common_path.FRONTEND_DIST_ROOT / "index.html").is_file()


def _resolve_client_file(full_path: str) -> Path | None:
    client_root = common_path.FRONTEND_DIST_ROOT.resolve()
    requested_path = (client_root / full_path).resolve()
    if not requested_path.is_relative_to(client_root):
        return None
    return requested_path if requested_path.is_file() else None


def _serve_client_root() -> FileResponse:
    return FileResponse(common_path.FRONTEND_DIST_ROOT / "index.html")


def _serve_client_path(full_path: str) -> FileResponse:
    client_file = _resolve_client_file(full_path)
    if client_file is not None:
        return FileResponse(client_file)
    return _serve_client_root()


def _redirect_root_to_docs() -> RedirectResponse:
    return RedirectResponse("/docs")


def create_app() -> FastAPI:
    settings = get_settings()
    application = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="Framework-independent visual model architecture service",
        lifespan=lifespan,
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    application.include_router(health.router)
    application.include_router(metadata.router)
    application.include_router(catalog.router)
    application.include_router(projects.router)
    application.include_router(validation.router)
    application.include_router(compilation.router)
    application.include_router(exports.router)

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


app = create_app()
