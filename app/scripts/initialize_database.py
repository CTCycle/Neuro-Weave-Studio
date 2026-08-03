from __future__ import annotations

import argparse
import logging
import re

from configurations.settings import get_database_settings
from repositories.initializer import initialize_database

LOGGER = logging.getLogger(__name__)


def _safe_error_message(error: Exception) -> str:
    message = str(error)
    message = re.sub(r"(?i)(postgres(?:ql)?(?:\+\w+)?://)[^@\s]+@", r"\1[redacted]@", message)
    return re.sub(r"(?i)(password\s*[=:]\s*)\S+", r"\1[redacted]", message)

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Explicitly initialize the configured PostgreSQL database and schema."
    )
    parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        initialize_database(get_database_settings())
    except Exception as exc:  # noqa: BLE001 - CLI boundary reports a sanitized failure
        LOGGER.error("Database initialization failed: %s", _safe_error_message(exc))
        return 1
    LOGGER.info("Database initialization completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
