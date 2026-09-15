from __future__ import annotations

from pydantic import BaseModel, Field


###############################################################################
class FilesystemEntry(BaseModel):
    name: str
    path: str
    is_dir: bool
    size_bytes: int | None = Field(default=None, ge=0)
    suffix: str | None = None

###############################################################################
class FilesystemBrowseResponse(BaseModel):
    current_path: str
    parent_path: str | None
    items: list[FilesystemEntry]
    roots: list[FilesystemEntry]
    truncated: bool = False
