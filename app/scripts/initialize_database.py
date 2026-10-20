# Copyright © 2026 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations

import argparse
import logging

from services.database_configuration_service import DatabaseConfigurationService

LOGGER = logging.getLogger(__name__)

###############################################################################
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Initialize or upgrade the configured database to the Alembic head."
    )
    parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    database_configuration = DatabaseConfigurationService()
    database_configuration.start()
    if not database_configuration.available:
        LOGGER.error("Database initialization failed: %s", database_configuration.error)
        database_configuration.dispose()
        return 1
    LOGGER.info("Database initialization completed")
    database_configuration.dispose()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
