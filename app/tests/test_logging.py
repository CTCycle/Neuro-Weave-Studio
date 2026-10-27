# Copyright © 2026 Thomas Virdis
# Licensed under the MIT License.

import logging

from common.logging_config import _AccessLogFilter, configure_logging


def test_access_filter_keeps_failures_and_debug_details(caplog):
    access_filter = _AccessLogFilter()

    def record(path, status):
        return logging.LogRecord("uvicorn.access", logging.INFO, "", 0, "%s %s %s %s %s", ("client", "GET", path, "1.1", status), None)

    with caplog.at_level(logging.INFO):
        assert not access_filter.filter(record("/health", 200))
        assert not access_filter.filter(record("/api/jobs/123", 404))
        assert access_filter.filter(record("/missing", 404))
    with caplog.at_level(logging.DEBUG):
        assert access_filter.filter(record("/health", 200))
        assert not access_filter.filter(record("/api/jobs/123", 200))


def test_configuration_updates_level_without_duplicate_filter():
    root = logging.getLogger()
    original = root.level
    try:
        configure_logging("DEBUG")
        configure_logging("WARNING")
        assert root.level == logging.WARNING
        assert sum(isinstance(item, _AccessLogFilter) for item in logging.getLogger("uvicorn.access").filters) == 1
    finally:
        root.setLevel(original)


def test_api_request_levels(client, caplog):
    with caplog.at_level(logging.DEBUG, logger="app"):
        assert client.get("/api/settings").status_code == 200
        assert client.get("/api/no-such-route").status_code == 404
    requests = [record for record in caplog.records if record.getMessage().startswith("api_request")]
    assert [(record.levelno, "status=200" in record.getMessage()) for record in requests] == [
        (logging.DEBUG, True), (logging.WARNING, False)
    ]
