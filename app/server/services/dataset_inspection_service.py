from __future__ import annotations

import ast
import csv
import struct
import zipfile
from collections import Counter
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import IO, Any, cast

import numpy as np
import pyarrow.parquet as parquet  # pyright: ignore[reportMissingTypeStubs]

from domain.datasets import (
    ArrayInspection,
    ArrayMetadata,
    ArraySource,
    ColumnMetadata,
    DatasetInspection,
    DatasetInspectionRequest,
    DatasetSourceInspection,
    ImageFolderInspection,
    ImageFolderSource,
    TableInspection,
    TableSource,
    TextInspection,
    TextSource,
)

IMAGE_EXTENSIONS = frozenset({".bmp", ".gif", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"})
TEXT_EXTENSIONS = frozenset({".jsonl", ".md", ".text", ".txt"})
TYPE_INFERENCE_LIMIT = 512
PARQUET = cast(Any, parquet)


###############################################################################
class DatasetInspectionError(ValueError):
    pass


###############################################################################
class DatasetInspectionService:
    """Read compact metadata from external dataset sources without copying sample data."""

    # -------------------------------------------------------------------------
    def inspect(self, request: DatasetInspectionRequest) -> DatasetInspection:
        source_ids = [source.id for source in request.sources]
        if len(source_ids) != len(set(source_ids)):
            raise DatasetInspectionError("Dataset source ids must be unique")

        inspections: list[DatasetSourceInspection] = []
        for source in request.sources:
            if isinstance(source, TableSource):
                inspections.append(self._inspect_table(source))
            elif isinstance(source, ImageFolderSource):
                inspections.append(self._inspect_images(source))
            elif isinstance(source, TextSource):
                inspections.append(self._inspect_text(source))
            else:
                inspections.append(self._inspect_array(source))

        counts = [item.sample_count for item in inspections]
        warnings: list[str] = []
        if len(set(counts)) > 1:
            warnings.append("Source sample counts differ; define and validate a relationship before registration.")
        return DatasetInspection(
            dataset_type=request.dataset_type,
            sample_count=min(counts) if counts else 0,
            sources=inspections,
            warnings=warnings,
        )

    # -------------------------------------------------------------------------
    def file_keys(self, source: ImageFolderSource | TextSource, mode: str) -> set[str]:
        root = self._validated_path(source.path, directory=source.type == "image_folder" or source.path_kind == "folder")
        paths = self._image_paths(root, source.recursive) if isinstance(source, ImageFolderSource) else self._text_paths(root, source)
        keys: set[str] = set()
        for path in paths:
            if mode == "filename":
                key = path.name
            elif mode == "stem":
                key = path.stem
            else:
                key = path.relative_to(root).as_posix() if root.is_dir() else path.name
            if key in keys:
                raise DatasetInspectionError(f"File mapping key is not unique: {key}")
            keys.add(key)
        return keys

    # -------------------------------------------------------------------------
    def table_identifiers(self, source: TableSource, column: str, mode: str) -> tuple[set[str], list[str]]:
        values: set[str] = set()
        duplicates: list[str] = []
        for raw in self._table_column_values(source, column):
            value = raw.strip()
            if not value:
                raise DatasetInspectionError(f"Table identifier column '{column}' contains an empty value")
            if mode == "filename":
                value = Path(value.replace("\\", "/")).name
            elif mode == "stem":
                value = Path(value.replace("\\", "/")).stem
            else:
                value = value.replace("\\", "/").removeprefix("./")
            if value in values and len(duplicates) < 10:
                duplicates.append(value)
            values.add(value)
        return values, duplicates

    # -------------------------------------------------------------------------
    def _inspect_table(self, source: TableSource) -> TableInspection:
        path = self._validated_path(source.path, directory=False)
        self._assert_suffix(path, source.format)
        if source.format == "parquet":
            try:
                dataset = PARQUET.ParquetFile(path)
                schema = dataset.schema_arrow
            except Exception as exc:
                raise DatasetInspectionError(f"Unable to read Parquet metadata: {exc}") from exc
            columns = [
                ColumnMetadata(name=field.name, dtype=str(field.type), nullable=field.nullable)
                for field in schema
            ]
            return TableInspection(
                source_id=source.id,
                path=str(path),
                format="parquet",
                sample_count=dataset.metadata.num_rows,
                columns=columns,
                inference_sample_size=0,
            )

        delimiter = source.delimiter or ("\t" if source.format == "tsv" else ",")
        try:
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.reader(handle, delimiter=delimiter, strict=True)
                first = next(reader, None)
                if first is None:
                    raise DatasetInspectionError("Table is empty")
                names = self._column_names(first, source.has_header)
                states = [_DTypeState() for _ in names]
                count = 0
                if not source.has_header:
                    self._consume_row(first, names, states)
                    count = 1
                for row in reader:
                    self._consume_row(row, names, states)
                    count += 1
                    if count > TYPE_INFERENCE_LIMIT:
                        states = [state.freeze() for state in states]
                states = [state.freeze() for state in states]
        except (OSError, UnicodeError, csv.Error) as exc:
            raise DatasetInspectionError(f"Unable to read delimited table: {exc}") from exc
        return TableInspection(
            source_id=source.id,
            path=str(path),
            format=source.format,
            sample_count=count,
            columns=[
                ColumnMetadata(name=name, dtype=state.dtype, nullable=state.nullable)
                for name, state in zip(names, states, strict=True)
            ],
            inference_sample_size=min(count, TYPE_INFERENCE_LIMIT),
        )

    # -------------------------------------------------------------------------
    def _inspect_images(self, source: ImageFolderSource) -> ImageFolderInspection:
        root = self._validated_path(source.path, directory=True)
        extensions: set[str] = set()
        class_counts: Counter[str] = Counter()
        count = 0
        for path in self._image_paths(root, source.recursive):
            count += 1
            extensions.add(path.suffix.casefold())
            if source.label_mode == "parent_folder":
                class_counts[path.parent.name] += 1
        return ImageFolderInspection(
            source_id=source.id,
            path=str(root),
            sample_count=count,
            extensions=sorted(extensions),
            class_names=sorted(class_counts),
            class_counts=dict(sorted(class_counts.items())),
        )

    # -------------------------------------------------------------------------
    def _inspect_text(self, source: TextSource) -> TextInspection:
        path = self._validated_path(source.path, directory=source.path_kind == "folder")
        sample_count = 0
        extensions: set[str] = set()
        for item in self._text_paths(path, source):
            sample_count += 1
            extensions.add(item.suffix.casefold())
        return TextInspection(
            source_id=source.id,
            path=str(path),
            sample_count=sample_count,
            extensions=sorted(extensions),
        )

    # -------------------------------------------------------------------------
    def _inspect_array(self, source: ArraySource) -> ArrayInspection:
        path = self._validated_path(source.path, directory=False)
        self._assert_suffix(path, source.format)
        try:
            if source.format == "npy":
                with path.open("rb") as handle:
                    shape, dtype = _read_npy_header(handle)
                arrays = [ArrayMetadata(key="array", shape=list(shape), dtype=dtype)]
            else:
                arrays: list[ArrayMetadata] = []
                with zipfile.ZipFile(path) as archive:
                    for name in sorted(archive.namelist()):
                        if not name.casefold().endswith(".npy"):
                            continue
                        with archive.open(name) as handle:
                            shape, dtype = _read_npy_header(handle)
                        arrays.append(
                            ArrayMetadata(
                                key=name[:-4],
                                shape=list(shape),
                                dtype=dtype,
                            )
                        )
        except (OSError, ValueError, zipfile.BadZipFile) as exc:
            raise DatasetInspectionError(f"Unable to read array metadata: {exc}") from exc
        if not arrays:
            raise DatasetInspectionError("Array source contains no NumPy arrays")
        leading_sizes = {item.shape[0] for item in arrays if item.shape}
        sample_count = min(leading_sizes) if leading_sizes else 1
        return ArrayInspection(
            source_id=source.id,
            path=str(path),
            format=source.format,
            sample_count=sample_count,
            arrays=arrays,
        )

    # -------------------------------------------------------------------------
    def _table_column_values(self, source: TableSource, column: str) -> Iterator[str]:
        path = self._validated_path(source.path, directory=False)
        if source.format == "parquet":
            try:
                parquet_file = PARQUET.ParquetFile(path)
                if column not in parquet_file.schema_arrow.names:
                    raise DatasetInspectionError(f"Column does not exist: {column}")
                for batch in parquet_file.iter_batches(columns=[column], batch_size=8192):
                    for value in batch.column(0).to_pylist():
                        yield "" if value is None else str(value)
            except DatasetInspectionError:
                raise
            except Exception as exc:
                raise DatasetInspectionError(f"Unable to read Parquet identifiers: {exc}") from exc
            return

        delimiter = source.delimiter or ("\t" if source.format == "tsv" else ",")
        try:
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.reader(handle, delimiter=delimiter, strict=True)
                first = next(reader, None)
                if first is None:
                    raise DatasetInspectionError("Table is empty")
                names = self._column_names(first, source.has_header)
                if column not in names:
                    raise DatasetInspectionError(f"Column does not exist: {column}")
                index = names.index(column)
                if not source.has_header:
                    yield first[index]
                for row in reader:
                    if len(row) != len(names):
                        raise DatasetInspectionError("Delimited table contains rows with inconsistent column counts")
                    yield row[index]
        except (OSError, UnicodeError, csv.Error) as exc:
            raise DatasetInspectionError(f"Unable to read delimited table identifiers: {exc}") from exc

    # -------------------------------------------------------------------------
    @staticmethod
    def _column_names(first: list[str], has_header: bool) -> list[str]:
        if not first:
            raise DatasetInspectionError("Table has no columns")
        names = [item.strip() for item in first] if has_header else [f"column_{index + 1}" for index in range(len(first))]
        if any(not name for name in names):
            raise DatasetInspectionError("Table header contains an empty column name")
        if len(names) != len(set(names)):
            raise DatasetInspectionError("Table header contains duplicate column names")
        return names

    # -------------------------------------------------------------------------
    @staticmethod
    def _consume_row(row: list[str], names: list[str], states: list[_DTypeState]) -> None:
        if len(row) != len(names):
            raise DatasetInspectionError("Delimited table contains rows with inconsistent column counts")
        for state, value in zip(states, row, strict=True):
            state.observe(value)

    # -------------------------------------------------------------------------
    @staticmethod
    def _validated_path(raw: str, *, directory: bool) -> Path:
        path = Path(raw).expanduser()
        if not path.exists():
            raise DatasetInspectionError(f"Path does not exist: {path}")
        if directory and not path.is_dir():
            raise DatasetInspectionError(f"Expected a folder: {path}")
        if not directory and not path.is_file():
            raise DatasetInspectionError(f"Expected a file: {path}")
        try:
            return path.resolve(strict=True)
        except OSError as exc:
            raise DatasetInspectionError(f"Unable to resolve path: {path}") from exc

    # -------------------------------------------------------------------------
    @staticmethod
    def _assert_suffix(path: Path, expected: str) -> None:
        if path.suffix.casefold() != f".{expected}":
            raise DatasetInspectionError(f"Expected a .{expected} file: {path}")

    # -------------------------------------------------------------------------
    @staticmethod
    def _image_paths(root: Path, recursive: bool) -> Iterator[Path]:
        iterator = root.rglob("*") if recursive else root.iterdir()
        try:
            for path in iterator:
                if path.is_file() and path.suffix.casefold() in IMAGE_EXTENSIONS:
                    yield path
        except PermissionError as exc:
            raise DatasetInspectionError(f"Permission denied while reading image folder: {root}") from exc

    # -------------------------------------------------------------------------
    @staticmethod
    def _text_paths(path: Path, source: TextSource) -> Iterator[Path]:
        if source.path_kind == "file":
            if path.suffix.casefold() not in TEXT_EXTENSIONS:
                raise DatasetInspectionError(f"Unsupported text file extension: {path.suffix or '(none)'}")
            yield path
            return
        iterator: Iterable[Path] = path.rglob("*") if source.recursive else path.iterdir()
        try:
            for item in iterator:
                if item.is_file() and item.suffix.casefold() in TEXT_EXTENSIONS:
                    yield item
        except PermissionError as exc:
            raise DatasetInspectionError(f"Permission denied while reading text folder: {path}") from exc


###############################################################################
class _DTypeState:

    # -------------------------------------------------------------------------
    def __init__(self) -> None:
        self.dtype = "unknown"
        self.nullable = False
        self._frozen = False

    # -------------------------------------------------------------------------
    def observe(self, raw: str) -> None:
        if self._frozen:
            return
        value = raw.strip()
        if not value:
            self.nullable = True
            return
        observed = _scalar_dtype(value)
        self.dtype = _merge_dtype(self.dtype, observed)

    # -------------------------------------------------------------------------
    def freeze(self) -> _DTypeState:
        self._frozen = True
        if self.dtype == "unknown":
            self.dtype = "string"
        return self


###############################################################################
def _scalar_dtype(value: str) -> str:
    if value.casefold() in {"true", "false"}:
        return "bool"
    try:
        int(value)
        return "int64"
    except ValueError:
        pass
    try:
        float(value)
        return "float64"
    except ValueError:
        return "string"


###############################################################################
def _merge_dtype(current: str, observed: str) -> str:
    if current in {"unknown", observed}:
        return observed
    if {current, observed} <= {"int64", "float64"}:
        return "float64"
    return "string"


###############################################################################
def _read_npy_header(handle: IO[bytes]) -> tuple[tuple[int, ...], str]:
    if handle.read(6) != b"\x93NUMPY":
        raise ValueError("Invalid NumPy array signature")
    version = handle.read(2)
    if len(version) != 2:
        raise ValueError("Truncated NumPy array header")
    major = version[0]
    length_bytes = handle.read(2 if major == 1 else 4)
    if len(length_bytes) not in {2, 4}:
        raise ValueError("Truncated NumPy array header length")
    header_length = struct.unpack("<H" if major == 1 else "<I", length_bytes)[0]
    header_bytes = handle.read(header_length)
    if len(header_bytes) != header_length:
        raise ValueError("Truncated NumPy array header")
    encoding = "utf-8" if major >= 3 else "latin1"
    metadata = cast(object, ast.literal_eval(header_bytes.decode(encoding).strip()))
    if not isinstance(metadata, dict):
        raise ValueError("Invalid NumPy array header metadata")
    values = cast(dict[object, object], metadata)
    shape = values.get("shape")
    descriptor = values.get("descr")
    if not isinstance(shape, tuple):
        raise ValueError("Invalid NumPy array shape")
    shape_values = cast(tuple[object, ...], shape)
    if not all(isinstance(item, int) and item >= 0 for item in shape_values):
        raise ValueError("Invalid NumPy array shape")
    return cast(tuple[int, ...], shape_values), str(np.dtype(cast(Any, descriptor)))
