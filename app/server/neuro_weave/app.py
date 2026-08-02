from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from neuro_weave.api.routes import (
    catalog,
    compilation,
    exports,
    health,
    metadata,
    projects,
    validation,
)
from neuro_weave.catalog.loader import get_node_catalog
from neuro_weave.config import get_settings
from neuro_weave.logging_config import configure_logging
from neuro_weave.persistence.session import close_repository_context, get_repository_context


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    settings.log_file.parent.mkdir(parents=True, exist_ok=True)
    configure_logging(settings.log_level, settings.log_file)
    settings.export_root.mkdir(parents=True, exist_ok=True)
    get_node_catalog()
    get_repository_context()
    try:
        yield
    finally:
        close_repository_context()


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
    return application


app = create_app()
