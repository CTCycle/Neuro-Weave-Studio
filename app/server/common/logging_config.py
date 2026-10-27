# Copyright © 2026 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations

import logging
from pathlib import Path


class _AccessLogFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        # Uvicorn passes (client, method, path, HTTP version, status).
        if not isinstance(record.args, tuple) or len(record.args) != 5:
            return True
        _, _, path, _, status = record.args
        path = str(path).split("?", 1)[0]
        if path == "/api" or path.startswith("/api/"):
            return False  # The application logs API responses with their duration.
        try:
            return int(status) >= 400 or logging.getLogger().isEnabledFor(logging.DEBUG)
        except (TypeError, ValueError):
            return True


###############################################################################
def configure_logging(level: str, log_file: Path | None = None) -> None:
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    root = logging.getLogger()
    root.setLevel(numeric_level)
    access_logger = logging.getLogger("uvicorn.access")
    if not any(isinstance(item, _AccessLogFilter) for item in access_logger.filters):
        access_logger.addFilter(_AccessLogFilter())
    if log_file is None:
        for handler in list(root.handlers):
            if getattr(handler, "_neuro_weave_log_file", None) is not None:
                root.removeHandler(handler)
                handler.close()
        return
    try:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        marker = str(log_file.resolve())
        for handler in list(root.handlers):
            configured = getattr(handler, "_neuro_weave_log_file", None)
            if configured is not None and configured != marker:
                root.removeHandler(handler)
                handler.close()
        if any(getattr(handler, "_neuro_weave_log_file", None) == marker for handler in root.handlers):
            for handler in root.handlers:
                if getattr(handler, "_neuro_weave_log_file", None) == marker:
                    handler.setLevel(numeric_level)
            return
        handler = logging.FileHandler(log_file, encoding="utf-8")
        handler._neuro_weave_log_file = marker  # type: ignore[attr-defined]
        handler.setLevel(numeric_level)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
        root.addHandler(handler)
    except OSError:
        logging.getLogger(__name__).warning("Could not open backend log file %s", log_file)
