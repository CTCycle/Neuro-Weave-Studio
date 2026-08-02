from __future__ import annotations

import argparse
import logging

from configurations.settings import get_database_settings
from repositories.initializer import initialize_database

LOGGER = logging.getLogger(__name__)

def main() -> int:
    parser = argparse.ArgumentParser(description="Initialize the NeuroWeave database schema.")
    parser.add_argument("--drop-existing", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        initialize_database(get_database_settings(), drop_existing=args.drop_existing)
    except Exception as exc:
        LOGGER.error("Database initialization failed: %s", exc)
        return 1
    LOGGER.info("Database initialization completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
