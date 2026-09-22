# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from services.filesystem_service import MAX_DIRECTORY_ENTRIES, FilesystemService


###############################################################################
def test_browse_lists_directories_before_files_and_filters_extensions(tmp_path: Path) -> None:
    (tmp_path / "images").mkdir()
    (tmp_path / "notes.txt").write_text("notes", encoding="utf-8")
    (tmp_path / "table.csv").write_text("x,y\n1,2\n", encoding="utf-8")

    response = FilesystemService().browse(
        str(tmp_path),
        include_files=True,
        extensions=(" CSV ",),
    )

    assert response.current_path == str(tmp_path.resolve())
    assert [item.name for item in response.items] == ["images", "table.csv"]
    assert response.items[0].is_dir is True
    assert response.items[1].suffix == ".csv"

###############################################################################
def test_browse_directory_only_omits_files(tmp_path: Path) -> None:
    (tmp_path / "child").mkdir()
    (tmp_path / "data.npy").write_bytes(b"array")

    response = FilesystemService().browse(str(tmp_path), include_files=False)

    assert [item.name for item in response.items] == ["child"]

###############################################################################
def test_browse_rejects_missing_path(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        FilesystemService().browse(str(tmp_path / "missing"))

###############################################################################
def test_browse_caps_large_directory_results(tmp_path: Path) -> None:
    for index in range(MAX_DIRECTORY_ENTRIES + 3):
        (tmp_path / f"folder-{index:04d}").mkdir()

    response = FilesystemService().browse(str(tmp_path), include_files=False)

    assert len(response.items) == MAX_DIRECTORY_ENTRIES
    assert response.truncated is True

###############################################################################
def test_filesystem_browse_api_returns_real_paths_and_filters_files(
    client: TestClient,
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "source"
    source_path.mkdir()
    (source_path / "nested").mkdir()
    (source_path / "keep.csv").write_text("value\n1\n", encoding="utf-8")
    (source_path / "skip.txt").write_text("skip", encoding="utf-8")

    response = client.get(
        "/api/filesystem/browse",
        params={"path": str(source_path), "include_files": "true", "extensions": "csv"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["current_path"] == str(source_path.resolve())
    assert [item["name"] for item in payload["items"]] == ["nested", "keep.csv"]
    assert payload["items"][0]["path"] == str((source_path / "nested").resolve())

###############################################################################
def test_filesystem_browse_api_rejects_missing_path(client: TestClient, tmp_path: Path) -> None:
    response = client.get(
        "/api/filesystem/browse",
        params={"path": str(tmp_path / "missing")},
    )

    assert response.status_code == 404
    assert "Path does not exist" in response.json()["detail"]
