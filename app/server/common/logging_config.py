from __future__ import annotations

import logging
from pathlib import Path

###############################################################################
def configure_logging(level: str, log_file: Path | None = None) -> None:
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    if log_file is None:
        return
    try:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        root = logging.getLogger()
        root.setLevel(numeric_level)
        marker = str(log_file.resolve())
        if any(getattr(handler, "_neuro_weave_log_file", None) == marker for handler in root.handlers):
            return
        handler = logging.FileHandler(log_file, encoding="utf-8")
        handler._neuro_weave_log_file = marker  # type: ignore[attr-defined]
        handler.setLevel(numeric_level)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
        root.addHandler(handler)
    except OSError:
        logging.getLogger(__name__).warning("Could not open backend log file %s", log_file)
