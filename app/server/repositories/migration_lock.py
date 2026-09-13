from __future__ import annotations

import logging
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

from filelock import FileLock, Timeout

LOGGER = logging.getLogger(__name__)

###############################################################################
@contextmanager
def database_migration_lock(db_path: Path, timeout: int) -> Generator[None]:
    """Serialize migrations for one database across processes and threads."""
    lock_path = Path(f"{db_path}.lock")
    lock = FileLock(str(lock_path), timeout=timeout)
    try:
        LOGGER.debug("Waiting for database migration lock")
        lock.acquire()
    except Timeout as exc:
        LOGGER.error(
            "Timed out after %ss waiting for the database migration lock",
            timeout,
        )
        raise RuntimeError(
            f"Timed out after {timeout}s waiting for the database migration lock. "
            "Another application instance may still be migrating."
        ) from exc
    try:
        LOGGER.debug("Acquired database migration lock")
        yield
    finally:
        lock.release()
        LOGGER.debug("Released database migration lock")
