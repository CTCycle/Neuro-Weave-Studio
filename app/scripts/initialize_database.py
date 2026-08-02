from __future__ import annotations

from common.logging_config import configure_logging
from configurations.settings import get_settings
from repositories.session import initialize_database

if __name__ == "__main__":
    settings = get_settings()
    configure_logging(settings.log_level)
    initialize_database()
