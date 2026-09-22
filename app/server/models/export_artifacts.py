from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any


###############################################################################
def stable_graph_payload(graph: Any) -> bytes:
    return json.dumps(
        graph.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

###############################################################################
def graph_sha256(graph: Any) -> str:
    return hashlib.sha256(stable_graph_payload(graph)).hexdigest()

###############################################################################
def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, sort_keys=True, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)

###############################################################################
def package_directory(
    directory: Path,
    destination: Path,
    *,
    archive_root: str,
    manifest: dict[str, Any],
) -> list[str]:
    if not directory.is_dir():
        raise RuntimeError(f"Expected export directory '{directory}'")
    write_json(directory / "manifest.json", manifest)
    files = [path for path in _relative_files(directory) if path.as_posix() != "manifest.json"]
    _write_zip(directory, destination, archive_root, files, manifest)
    return ["manifest.json", *[f"{archive_root}/{path.as_posix()}" for path in files]]

###############################################################################
def package_files(
    files: Iterable[tuple[Path, str]],
    destination: Path,
    *,
    manifest: dict[str, Any],
) -> list[str]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    normalized = [(source, _safe_archive_path(name)) for source, name in files]
    names = [name for _, name in normalized]
    if len(names) != len(set(names)):
        raise RuntimeError("Export package contains duplicate file names")
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(_zip_info("manifest.json"), json.dumps(manifest, sort_keys=True, indent=2))
        for source, name in sorted(normalized, key=lambda item: item[1]):
            if not source.is_file():
                raise RuntimeError(f"Export package file is missing: {source.name}")
            archive.writestr(_zip_info(name), source.read_bytes())
    temporary.replace(destination)
    return ["manifest.json", *names]

###############################################################################
def artifact_file_metadata(path: Path, *, relative_path: str | None = None) -> dict[str, object]:
    if not path.is_file():
        raise RuntimeError(f"Artifact file is missing: {path}")
    return {
        "path": relative_path or path.name,
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "media_type": _media_type(path),
    }

###############################################################################
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

###############################################################################
def remove_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    else:
        path.unlink(missing_ok=True)

###############################################################################
def _relative_files(directory: Path) -> list[Path]:
    files: list[Path] = []
    for path in directory.rglob("*"):
        if path.is_symlink():
            raise RuntimeError("Symlinks are not allowed in export packages")
        if path.is_file():
            files.append(path.relative_to(directory))
    return files

###############################################################################
def _write_zip(
    directory: Path,
    destination: Path,
    archive_root: str,
    files: list[Path],
    manifest: dict[str, Any],
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(_zip_info("manifest.json"), json.dumps(manifest, sort_keys=True, indent=2))
        for relative in sorted(files, key=lambda item: item.as_posix()):
            archive_name = _safe_archive_path(f"{archive_root}/{relative.as_posix()}")
            archive.writestr(_zip_info(archive_name), (directory / relative).read_bytes())
    temporary.replace(destination)

###############################################################################
def _safe_archive_path(value: str) -> str:
    normalized = value.replace("\\", "/")
    path = Path(normalized)
    if path.is_absolute() or ".." in path.parts:
        raise RuntimeError(f"Unsafe export archive path '{value}'")
    return "/".join(part for part in path.parts if part not in {"", "."})

###############################################################################
def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(filename=name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 0
    return info

###############################################################################
def _media_type(path: Path) -> str:
    return {
        ".json": "application/json",
        ".onnx": "application/vnd.nvidia.onnx",
        ".py": "text/x-python",
        ".zip": "application/zip",
        ".keras": "application/octet-stream",
        ".h5": "application/octet-stream",
        ".pt2": "application/octet-stream",
        ".safetensors": "application/octet-stream",
        ".msgpack": "application/octet-stream",
    }.get(path.suffix.lower(), "application/octet-stream")
