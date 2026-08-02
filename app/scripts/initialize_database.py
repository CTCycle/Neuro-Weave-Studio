from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

SERVER_ROOT = Path(__file__).resolve().parents[1] / "server"
if str(SERVER_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVER_ROOT))

from neuro_weave.config import get_database_settings
from neuro_weave.persistence.initializer import initialize_database

LOGGER = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="Initialize the NeuroWeave database schema.")
    parser.add_argument(
        "--drop-existing",
        action="store_true",
        help="Delete the embedded SQLite file or drop NeuroWeave tables before recreating them.",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        initialize_database(get_database_settings(), drop_existing=args.drop_existing)
    except Exception as exc:  # noqa: BLE001 - CLI must report every startup failure safely
        LOGGER.error("Database initialization failed: %s", exc)
        return 1
    LOGGER.info("Database initialization completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
