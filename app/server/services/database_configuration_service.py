# Copyright © 2026 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

from configurations.database_config import (
    DATABASE_CONFIG_PATH,
    DatabaseConfig,
    bootstrap_database_config,
    write_database_config,
)
from configurations.settings import Settings
from repositories.session import RepositoryContext, create_repository_context

LOGGER = logging.getLogger(__name__)

###############################################################################
@dataclass(frozen=True, slots=True)
class DatabaseStatus:
    backend: str
    sqlite_path: str | None
    connect_timeout_seconds: int | None
    available: bool
    switching: bool
    message: str | None

###############################################################################
class DatabaseSwitchBusyError(RuntimeError):
    pass

###############################################################################
class DatabaseConfigurationService:
    """Owns the active database context, its settings, and request-switch barrier."""

    _REQUEST_DRAIN_TIMEOUT_SECONDS = 30

    # -------------------------------------------------------------------------
    def __init__(
        self,
        config_path: Path = DATABASE_CONFIG_PATH,
        *,
        initial_settings: Settings | None = None,
    ) -> None:
        self.config_path = Path(config_path)
        self.initial_settings = initial_settings
        self.config: DatabaseConfig | None = None
        self.repository_context: RepositoryContext | None = None
        self.settings = Settings()
        self.error: str | None = None
        self._condition = asyncio.Condition()
        self._configuration_operation_lock = asyncio.Lock()
        self._active_requests = 0
        self._switching = False

    # -------------------------------------------------------------------------
    @property
    def available(self) -> bool:
        return self.repository_context is not None

    # -------------------------------------------------------------------------
    @property
    def switching(self) -> bool:
        return self._switching

    # -------------------------------------------------------------------------
    def start(self) -> None:
        context: RepositoryContext | None = None
        try:
            config, config_was_missing = bootstrap_database_config(self.config_path)
            self.config = config
            context = create_repository_context(config.to_database_settings())
            settings = self._load_or_seed_settings(context, self.initial_settings)
            if config_was_missing:
                write_database_config(config, self.config_path)
            self.repository_context = context
            self.settings = settings
            self.error = None
        except Exception as error:  # startup remains available for database repair
            if context is not None and context is not self.repository_context:
                context.dispose()
            self.error = str(error)

    # -------------------------------------------------------------------------
    def dispose(self) -> None:
        context = self.repository_context
        self.repository_context = None
        if context is not None:
            context.dispose()

    # -------------------------------------------------------------------------
    def status(self) -> DatabaseStatus:
        config = self.config or DatabaseConfig()
        return DatabaseStatus(
            backend=config.backend,
            sqlite_path=str(config.sqlite_path),
            connect_timeout_seconds=config.connect_timeout_seconds,
            available=self.available,
            switching=self._switching,
            message=self.error,
        )

    # -------------------------------------------------------------------------
    async def enter_request(self) -> None:
        async with self._condition:
            await self._condition.wait_for(lambda: not self._switching)
            self._active_requests += 1

    # -------------------------------------------------------------------------
    async def leave_request(self) -> None:
        async with self._condition:
            self._active_requests = max(0, self._active_requests - 1)
            self._condition.notify_all()

    # -------------------------------------------------------------------------
    @asynccontextmanager
    async def pause_requests(self) -> AsyncGenerator[None]:
        async with self._condition:
            await self._condition.wait_for(lambda: not self._switching)
            self._switching = True
            try:
                await asyncio.wait_for(
                    self._condition.wait_for(lambda: self._active_requests == 0),
                    timeout=self._REQUEST_DRAIN_TIMEOUT_SECONDS,
                )
            except TimeoutError as error:
                self._switching = False
                self._condition.notify_all()
                raise DatabaseSwitchBusyError(
                    "Database operations are still in progress. Wait briefly and retry."
                ) from error
        try:
            yield
        finally:
            async with self._condition:
                self._switching = False
                self._condition.notify_all()

    # -------------------------------------------------------------------------
    async def test(self, candidate: DatabaseConfig) -> None:
        async with self._configuration_operation_lock:
            if self.config is not None and candidate.sqlite_path == self.config.sqlite_path:
                async with self.pause_requests():
                    context = await asyncio.to_thread(self._prepare_context, candidate)
            else:
                context = await asyncio.to_thread(self._prepare_context, candidate)
            context.dispose()

    # -------------------------------------------------------------------------
    async def activate(
        self,
        candidate: DatabaseConfig,
        *,
        has_active_jobs: Callable[[], bool],
        apply_settings: Callable[[Settings], None],
    ) -> Settings:
        async with self._configuration_operation_lock, self.pause_requests():
            if has_active_jobs():
                raise DatabaseSwitchBusyError(
                    "Stop or wait for active runtime jobs before switching databases."
                )
            context, settings = await asyncio.to_thread(
                self._prepare_context_with_settings, candidate
            )
            previous_context = self.repository_context
            previous_settings = self.settings
            try:
                apply_settings(settings)
                await asyncio.to_thread(write_database_config, candidate, self.config_path)
            except Exception:
                try:
                    apply_settings(previous_settings)
                finally:
                    context.dispose()
                raise
            self.config = candidate
            self.repository_context = context
            self.settings = settings
            self.error = None
            if previous_context is not None:
                try:
                    previous_context.dispose()
                except Exception:
                    LOGGER.exception("Could not dispose the previous database engine")
            return settings

    # -------------------------------------------------------------------------
    @staticmethod
    def _prepare_context(candidate: DatabaseConfig) -> RepositoryContext:
        return create_repository_context(candidate.to_database_settings())

    # -------------------------------------------------------------------------
    def _prepare_context_with_settings(
        self,
        candidate: DatabaseConfig,
    ) -> tuple[RepositoryContext, Settings]:
        context = self._prepare_context(candidate)
        try:
            return context, self._load_or_seed_settings(context, None)
        except Exception:
            context.dispose()
            raise

    # -------------------------------------------------------------------------
    @staticmethod
    def _load_or_seed_settings(
        context: RepositoryContext,
        initial_settings: Settings | None,
    ) -> Settings:
        with context.new_unit_of_work() as uow:
            settings = uow.application_settings.get()
            if settings is None:
                settings = initial_settings or Settings()
                uow.application_settings.save(settings)
                uow.commit()
            return settings
