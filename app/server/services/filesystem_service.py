from __future__ import annotations

import os
from pathlib import Path

from domain.filesystem import FilesystemBrowseResponse, FilesystemEntry

MAX_DIRECTORY_ENTRIES = 500


###############################################################################
class FilesystemService:
    """Read-only local filesystem navigation for path-backed workflows."""

    # -------------------------------------------------------------------------
    def browse(
        self,
        path: str = "",
        *,
        include_files: bool = True,
        extensions: tuple[str, ...] = (),
    ) -> FilesystemBrowseResponse:
        roots = self._roots()
        normalized_extensions = tuple(
            normalized if normalized.startswith(".") else f".{normalized}"
            for extension in extensions
            if (normalized := extension.strip().casefold())
        )
        if not path.strip():
            return FilesystemBrowseResponse(
                current_path="",
                parent_path=None,
                items=roots,
                roots=roots,
            )

        directory = Path(path.strip()).expanduser()
        if not directory.exists():
            raise FileNotFoundError(f"Path does not exist: {directory}")
        if not directory.is_dir():
            raise NotADirectoryError(f"Path is not a directory: {directory}")

        directory = directory.resolve()
        parent = directory.parent
        parent_path = "" if parent == directory else str(parent)
        entries: list[FilesystemEntry] = []
        truncated = False
        try:
            children = sorted(
                directory.iterdir(),
                key=lambda item: (not item.is_dir(), item.name.casefold()),
            )
            for child in children:
                is_dir = child.is_dir()
                if not is_dir:
                    if not include_files:
                        continue
                    if normalized_extensions and child.suffix.casefold() not in normalized_extensions:
                        continue
                if len(entries) >= MAX_DIRECTORY_ENTRIES:
                    truncated = True
                    break
                size_bytes = None
                if not is_dir:
                    try:
                        size_bytes = child.stat().st_size
                    except OSError:
                        size_bytes = None
                entries.append(
                    FilesystemEntry(
                        name=child.name,
                        path=str(child),
                        is_dir=is_dir,
                        size_bytes=size_bytes,
                        suffix=None if is_dir else child.suffix.casefold() or None,
                    )
                )
        except PermissionError as exc:
            raise PermissionError(f"Permission denied: {directory}") from exc

        return FilesystemBrowseResponse(
            current_path=str(directory),
            parent_path=parent_path,
            items=entries,
            roots=roots,
            truncated=truncated,
        )

    # -------------------------------------------------------------------------
    def _roots(self) -> list[FilesystemEntry]:
        candidates: list[Path] = []
        if os.name == "nt":
            for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
                drive = Path(f"{letter}:\\")
                if drive.exists():
                    candidates.append(drive)
        else:
            candidates.extend([Path.home(), Path("/")])

        roots: list[FilesystemEntry] = []
        seen: set[str] = set()
        for candidate in candidates:
            try:
                resolved = str(candidate.resolve())
            except OSError:
                resolved = str(candidate)
            if resolved in seen or not candidate.exists() or not candidate.is_dir():
                continue
            seen.add(resolved)
            roots.append(
                FilesystemEntry(
                    name=resolved,
                    path=resolved,
                    is_dir=True,
                )
            )
        return roots
