from __future__ import annotations

import shutil
from pathlib import Path

from common.path import ENV_EXAMPLE_FILE, ENV_FILE


###############################################################################
def ensure_environment_file(
    environment_file: Path = ENV_FILE,
    environment_example_file: Path = ENV_EXAMPLE_FILE,
) -> Path:
    """Create the local environment file from its template only when absent."""
    target = Path(environment_file)
    template = Path(environment_example_file)

    if target.exists():
        if not target.is_file():
            raise IsADirectoryError(f"Environment path is not a file: {target}")
        return target
    if not template.is_file():
        raise FileNotFoundError(f"Missing environment template: {template}")

    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with target.open("xb") as destination, template.open("rb") as source:
            shutil.copyfileobj(source, destination)
    except FileExistsError:
        # Another startup created the file after the existence check.
        pass
    return target
