# Copyright © 2026 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations

from datetime import UTC, datetime

###############################################################################
def utc_now() -> datetime:
    return datetime.now(UTC)
