from __future__ import annotations

# isort: skip_file

import sys
from pathlib import Path

SERVER_ROOT = Path(__file__).resolve().parents[1] / "server"
if str(SERVER_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVER_ROOT))

from configurations.environment import ensure_environment_file


if __name__ == "__main__":
    ensure_environment_file()
