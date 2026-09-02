from __future__ import annotations

from pathlib import Path

import pytest

from services.filesystem_service import MAX_DIRECTORY_ENTRIES, FilesystemService


def test_browse_lists_directories_before_files_and_filters_extensions(tmp_path: Path) -> None:
    (tmp_path / "images").mkdir()
    (tmp_path / "notes.txt").write_text("notes", encoding="utf-8")
    (tmp_path / "table.csv").write_text("x,y\n1,2\n", encoding="utf-8")

    response = FilesystemService().browse(
        str(tmp_path),
        include_files=True,
        extensions=("csv",),
    )

    assert response.current_path == str(tmp_path.resolve())
    assert [item.name for item in response.items] == ["images", "table.csv"]
    assert response.items[0].is_dir is True
    assert response.items[1].suffix == ".csv"


def test_browse_directory_only_omits_files(tmp_path: Path) -> None:
    (tmp_path / "child").mkdir()
    (tmp_path / "data.npy").write_bytes(b"array")

    response = FilesystemService().browse(str(tmp_path), include_files=False)

    assert [item.name for item in response.items] == ["child"]


def test_browse_rejects_missing_path(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        FilesystemService().browse(str(tmp_path / "missing"))


def test_browse_caps_large_directory_results(tmp_path: Path) -> None:
    for index in range(MAX_DIRECTORY_ENTRIES + 3):
        (tmp_path / f"folder-{index:04d}").mkdir()

    response = FilesystemService().browse(str(tmp_path), include_files=False)

    assert len(response.items) == MAX_DIRECTORY_ENTRIES
    assert response.truncated is True
