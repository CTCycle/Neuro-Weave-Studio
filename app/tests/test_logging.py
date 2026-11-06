# Copyright © 2026 Thomas Virdis
# Licensed under the MIT License.

import logging

import pytest
from common.logging_config import AccessLogFilter, configure_logging
from fastapi.testclient import TestClient


def _record(path: str, status: int) -> logging.LogRecord:
    return logging.LogRecord("uvicorn.access", logging.INFO, "", 0, "%s %s %s %s %s", ("client", "GET", path, "1.1", status), None)


def test_access_filter_keeps_failures_and_debug_details(caplog: pytest.LogCaptureFixture) -> None:
    access_filter = AccessLogFilter()

    with caplog.at_level(logging.INFO):
        assert not access_filter.filter(_record("/health", 200))
        assert not access_filter.filter(_record("/api/jobs/123", 404))
        assert access_filter.filter(_record("/missing", 404))
    with caplog.at_level(logging.DEBUG):
        assert access_filter.filter(_record("/health", 200))
        assert not access_filter.filter(_record("/api/jobs/123", 200))


def test_configuration_updates_level_without_duplicate_filter() -> None:
    root = logging.getLogger()
    original = root.level
    try:
        configure_logging("DEBUG")
        configure_logging("WARNING")
        assert root.level == logging.WARNING
        assert sum(isinstance(item, AccessLogFilter) for item in logging.getLogger("uvicorn.access").filters) == 1
    finally:
        root.setLevel(original)


def test_api_request_levels(client: TestClient, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.DEBUG, logger="app"):
        assert client.get("/api/settings").status_code == 200
        assert client.get("/api/no-such-route").status_code == 404
    requests = [record for record in caplog.records if record.getMessage().startswith("api_request")]
    assert [(record.levelno, "status=200" in record.getMessage()) for record in requests] == [
        (logging.DEBUG, True), (logging.WARNING, False)
    ]


def test_lifespan_disposes_database_when_startup_fails_before_controller(monkeypatch: pytest.MonkeyPatch) -> None:
    import asyncio
    from types import SimpleNamespace
    from typing import Any

    import app as app_module

    disposed: list[bool] = []

    class FailingConfiguration:
        settings = SimpleNamespace(log_level="INFO", log_file=None, export_root=None)
        repository_context = None

        def start(self) -> None:
            raise RuntimeError("boot failed")

        def dispose(self) -> None:
            disposed.append(True)

    application: Any = SimpleNamespace(state=SimpleNamespace(database_configuration=FailingConfiguration()))

    async def run() -> None:
        async with app_module.lifespan(application):
            pass

    with pytest.raises(RuntimeError, match="boot failed"):
        asyncio.run(run())
    assert disposed == [True]
