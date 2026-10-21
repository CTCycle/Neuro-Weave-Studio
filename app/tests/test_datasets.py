# Copyright © 2026 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable
from importlib import import_module
from pathlib import Path
from typing import Protocol, cast

import numpy as np
from fastapi import FastAPI
from fastapi.testclient import TestClient

###############################################################################
class _PyArrowModule(Protocol):

    # -------------------------------------------------------------------------
    def table(self, data: object) -> object: ...

###############################################################################
class _ParquetModule(Protocol):

    # -------------------------------------------------------------------------
    def write_table(self, table: object, where: Path) -> None: ...


pa = cast(_PyArrowModule, import_module("pyarrow"))
parquet = cast(_ParquetModule, import_module("pyarrow.parquet"))

###############################################################################
def _table_source(path: Path, source_id: str = "table") -> dict[str, object]:
    return {
        "type": "table",
        "id": source_id,
        "path": str(path),
        "format": "csv",
        "has_header": True,
        "delimiter": ",",
    }

###############################################################################
def _tabular_payload(path: Path, *, subset: dict[str, object] | None = None) -> dict[str, object]:
    return {
        "name": "Patient outcomes",
        "dataset_type": "tabular",
        "sources": [_table_source(path)],
        "input_definition": {
            "bindings": [
                {"type": "columns", "source_id": "table", "columns": ["age", "weight"]}
            ]
        },
        "output_definition": {
            "bindings": [{"type": "columns", "source_id": "table", "columns": ["diagnosis"]}]
        },
        "metadata_definition": {"bindings": []},
        "relationships": [],
        "sequence_definition": None,
        "subset_definition": subset or {"type": "all"},
    }

###############################################################################
def test_tabular_registration_persists_metadata_only_and_lists_summary(
    client: TestClient, tmp_path: Path
) -> None:
    source = tmp_path / "patients.csv"
    source.write_text(
        "age,weight,diagnosis\n42,71,positive\n50,80,negative\n",
        encoding="utf-8",
    )

    created = client.post("/api/datasets", json=_tabular_payload(source))

    assert created.status_code == 201
    document = created.json()
    assert document["sample_count"] == 2
    assert document["selected_sample_count"] == 2
    assert document["sources"][0]["path"] == str(source.resolve())
    assert document["inspection"]["sources"][0]["columns"][0]["dtype"] == "int64"

    listed = client.get("/api/datasets")
    assert listed.status_code == 200
    assert listed.json()["datasets"][0]["input_summary"] == "2 columns"
    assert listed.json()["datasets"][0]["output_summary"] == "1 column"

    raw = sqlite3.connect(tmp_path / "test.db").execute(
        "SELECT definition_json, inspection_json, validation_json FROM datasets"
    ).fetchone()
    assert raw is not None
    serialized = json.dumps(raw)
    assert "positive" not in serialized
    assert "negative" not in serialized

###############################################################################
def test_validate_rejects_missing_columns_and_create_returns_structured_findings(
    client: TestClient, tmp_path: Path
) -> None:
    source = tmp_path / "patients.csv"
    source.write_text("age,diagnosis\n42,positive\n", encoding="utf-8")
    payload = _tabular_payload(source)
    payload["input_definition"] = {
        "bindings": [{"type": "columns", "source_id": "table", "columns": ["missing"]}]
    }

    validation = client.post("/api/datasets/validate", json=payload)
    assert validation.status_code == 200
    validation_payload = validation.json()
    assert validation_payload["valid"] is False
    assert any(item["code"] == "missing_columns" for item in validation_payload["validation"]["errors"])

    created = client.post("/api/datasets", json=payload)
    assert created.status_code == 422
    detail = created.json()["detail"]
    assert detail["code"] == "dataset_definition_invalid"
    registration_validation = detail["validation"]
    validation_payload["validation"]["checked_at"] = registration_validation["validation"]["checked_at"]
    assert registration_validation == validation_payload

###############################################################################
def test_invalid_update_returns_complete_validation_without_mutating_registration(
    client: TestClient, tmp_path: Path
) -> None:
    source = tmp_path / "patients.csv"
    source.write_text("age,weight,diagnosis\n42,71,positive\n", encoding="utf-8")
    created = client.post("/api/datasets", json=_tabular_payload(source))
    assert created.status_code == 201
    dataset_id = created.json()["id"]

    payload = _tabular_payload(source)
    payload["name"] = "Rejected update"
    payload["input_definition"] = {
        "bindings": [{"type": "columns", "source_id": "table", "columns": ["missing"]}]
    }
    validation = client.post("/api/datasets/validate", json=payload)
    rejected = client.put(f"/api/datasets/{dataset_id}", json=payload)

    assert validation.status_code == 200
    assert rejected.status_code == 422
    detail = rejected.json()["detail"]
    assert detail["code"] == "dataset_definition_invalid"
    registration_validation = detail["validation"]
    validation_payload = validation.json()
    validation_payload["validation"]["checked_at"] = registration_validation["validation"]["checked_at"]
    assert registration_validation == validation_payload
    assert client.get(f"/api/datasets/{dataset_id}").json()["name"] == "Patient outcomes"

###############################################################################
def test_update_keeps_registered_dataset_type_immutable(client: TestClient, tmp_path: Path) -> None:
    source = tmp_path / "type-stable.csv"
    source.write_text("age,weight,diagnosis\n42,71,positive\n", encoding="utf-8")
    created = client.post("/api/datasets", json=_tabular_payload(source))
    assert created.status_code == 201
    dataset_id = created.json()["id"]

    changed = _tabular_payload(source)
    changed["dataset_type"] = "time_series"
    changed["name"] = "Changed type"
    rejected = client.put(f"/api/datasets/{dataset_id}", json=changed)

    assert rejected.status_code == 422
    errors = rejected.json()["detail"]["validation"]["validation"]["errors"]
    assert any(item["code"] == "dataset_type_immutable" for item in errors)
    persisted = client.get(f"/api/datasets/{dataset_id}").json()
    assert persisted["dataset_type"] == "tabular"
    assert persisted["name"] == "Patient outcomes"

###############################################################################
def test_text_folder_inspection_streams_exact_count_and_extensions(
    client: TestClient, tmp_path: Path
) -> None:
    source = tmp_path / "text"
    nested = source / "nested"
    nested.mkdir(parents=True)
    (source / "first.TXT").write_text("first", encoding="utf-8")
    (source / "second.md").write_text("second", encoding="utf-8")
    (nested / "third.jsonl").write_text("{}", encoding="utf-8")
    (nested / "ignored.csv").write_text("value\n1\n", encoding="utf-8")

    response = client.post(
        "/api/datasets/inspect",
        json={
            "dataset_type": "text",
            "sources": [
                {
                    "type": "text",
                    "id": "text",
                    "path": str(source),
                    "path_kind": "folder",
                    "recursive": True,
                }
            ],
        },
    )

    assert response.status_code == 200
    inspection = response.json()
    assert inspection["sample_count"] == 3
    assert inspection["sources"][0]["sample_count"] == 3
    assert inspection["sources"][0]["extensions"] == [".jsonl", ".md", ".txt"]

###############################################################################
def test_text_file_registration_and_text_table_composed_mapping(
    client: TestClient, tmp_path: Path
) -> None:
    text_file = tmp_path / "notes.jsonl"
    text_file.write_text('{"text":"one"}\n', encoding="utf-8")
    text_payload: dict[str, object] = {
        "name": "Text file",
        "dataset_type": "text",
        "sources": [
            {
                "type": "text",
                "id": "text",
                "path": str(text_file),
                "path_kind": "file",
                "recursive": False,
            }
        ],
        "input_definition": {"bindings": [{"type": "text", "source_id": "text"}]},
        "output_definition": {"bindings": []},
        "metadata_definition": {"bindings": []},
        "relationships": [],
        "sequence_definition": None,
        "subset_definition": {"type": "all"},
    }

    text_created = client.post("/api/datasets", json=text_payload)

    assert text_created.status_code == 201
    assert text_created.json()["sample_count"] == 1
    assert text_created.json()["inspection"]["sources"][0]["extensions"] == [".jsonl"]

    text_folder = tmp_path / "documents"
    text_folder.mkdir()
    (text_folder / "first.md").write_text("first", encoding="utf-8")
    (text_folder / "second.txt").write_text("second", encoding="utf-8")
    table = tmp_path / "document-labels.csv"
    table.write_text("filename,label\nfirst.md,one\nsecond.txt,two\n", encoding="utf-8")
    composed_payload: dict[str, object] = {
        "name": "Text labels",
        "dataset_type": "composed",
        "sources": [
            {
                "type": "text",
                "id": "documents",
                "path": str(text_folder),
                "path_kind": "folder",
                "recursive": True,
            },
            _table_source(table, "labels"),
        ],
        "input_definition": {
            "bindings": [
                {"type": "text", "source_id": "documents"},
                {"type": "columns", "source_id": "labels", "columns": ["label"]},
            ]
        },
        "output_definition": {"bindings": []},
        "metadata_definition": {"bindings": []},
        "relationships": [
            {
                "type": "file_table",
                "file_source_id": "documents",
                "file_key": "filename",
                "table_source_id": "labels",
                "table_key_column": "filename",
            }
        ],
        "sequence_definition": None,
        "subset_definition": {"type": "all"},
    }

    composed_created = client.post("/api/datasets", json=composed_payload)

    assert composed_created.status_code == 201
    assert composed_created.json()["sample_count"] == 2

###############################################################################
def test_tsv_time_series_registration_preserves_structural_columns(
    client: TestClient, tmp_path: Path
) -> None:
    source = tmp_path / "events.tsv"
    source.write_text(
        "timestamp\tgroup\tvalue\tlabel\n2026-01-01\ta\t1\tlow\n2026-01-02\ta\t2\thigh\n",
        encoding="utf-8",
    )
    payload = _tabular_payload(source)
    payload["name"] = "Events"
    payload["dataset_type"] = "time_series"
    payload["sources"] = [{**_table_source(source), "format": "tsv", "delimiter": "\t"}]
    payload["input_definition"] = {
        "bindings": [{"type": "columns", "source_id": "table", "columns": ["value"]}]
    }
    payload["output_definition"] = {
        "bindings": [{"type": "columns", "source_id": "table", "columns": ["label"]}]
    }
    payload["sequence_definition"] = {"timestamp_column": "timestamp", "group_column": "group"}

    created = client.post("/api/datasets", json=payload)

    assert created.status_code == 201
    assert created.json()["format"] == "TSV"
    assert created.json()["sample_count"] == 2
    assert created.json()["sequence_definition"] == payload["sequence_definition"]

###############################################################################
def test_source_extension_mismatch_is_blocking(
    client: TestClient, tmp_path: Path
) -> None:
    source = tmp_path / "patients.csv"
    source.write_text("age,diagnosis\n42,positive\n", encoding="utf-8")
    payload = _tabular_payload(source)
    payload["sources"] = [{**_table_source(source), "format": "tsv", "delimiter": "\t"}]

    validation = client.post("/api/datasets/validate", json=payload)

    assert validation.status_code == 200
    assert validation.json()["valid"] is False
    assert any(
        item["code"] == "source_inspection_failed"
        for item in validation.json()["validation"]["errors"]
    )

###############################################################################
def test_first_n_subset_is_persisted_and_out_of_bounds_is_blocking(
    client: TestClient, tmp_path: Path
) -> None:
    source = tmp_path / "patients.csv"
    source.write_text(
        "age,weight,diagnosis\n42,71,positive\n50,80,negative\n61,90,positive\n",
        encoding="utf-8",
    )

    payload = _tabular_payload(source, subset={"type": "first_n", "count": 2})
    created = client.post("/api/datasets", json=payload)
    assert created.status_code == 201
    assert created.json()["selected_sample_count"] == 2

    invalid = _tabular_payload(source, subset={"type": "first_n", "count": 4})
    validation = client.post("/api/datasets/validate", json=invalid)
    assert validation.json()["valid"] is False
    assert any(item["code"] == "subset_out_of_bounds" for item in validation.json()["validation"]["errors"])

###############################################################################
def test_image_folder_and_label_table_require_exact_mapping(
    client: TestClient, tmp_path: Path
) -> None:
    images = tmp_path / "images"
    images.mkdir()
    (images / "img 1.jpg").write_bytes(b"not decoded")
    (images / "img-2.png").write_bytes(b"not decoded")
    labels = tmp_path / "labels.csv"
    labels.write_text(
        "filename,label\nimg 1.jpg,cat\nimg-2.png,dog\n",
        encoding="utf-8",
    )
    payload: dict[str, object] = {
        "name": "Image labels",
        "dataset_type": "images",
        "sources": [
            {
                "type": "image_folder",
                "id": "images",
                "path": str(images),
                "recursive": True,
                "label_mode": "none",
            },
            _table_source(labels, "labels"),
        ],
        "input_definition": {
            "bindings": [{"type": "image", "source_id": "images"}]
        },
        "output_definition": {
            "bindings": [{"type": "columns", "source_id": "labels", "columns": ["label"]}]
        },
        "metadata_definition": {"bindings": []},
        "relationships": [
            {
                "type": "file_table",
                "file_source_id": "images",
                "file_key": "filename",
                "table_source_id": "labels",
                "table_key_column": "filename",
            }
        ],
        "sequence_definition": None,
        "subset_definition": {"type": "all"},
    }

    created = client.post("/api/datasets", json=payload)
    assert created.status_code == 201
    assert created.json()["sample_count"] == 2

    labels.write_text("filename,label\nimg 1.jpg,cat\nmissing.png,dog\n", encoding="utf-8")
    invalid = client.post("/api/datasets/validate", json=payload)
    assert invalid.json()["valid"] is False
    assert any(item["code"] == "files_without_rows" for item in invalid.json()["validation"]["errors"])
    assert any(item["code"] == "rows_without_files" for item in invalid.json()["validation"]["errors"])

###############################################################################
def test_array_sources_validate_leading_sample_dimensions(
    client: TestClient, tmp_path: Path
) -> None:
    features = tmp_path / "X.npy"
    targets = tmp_path / "y.npy"
    np.save(features, np.ones((4, 3), dtype=np.float32))
    np.save(targets, np.arange(4, dtype=np.int64))
    payload: dict[str, object] = {
        "name": "Array regression",
        "dataset_type": "array",
        "sources": [
            {"type": "array", "id": "features", "path": str(features), "format": "npy"},
            {"type": "array", "id": "targets", "path": str(targets), "format": "npy"},
        ],
        "input_definition": {
            "bindings": [{"type": "array", "source_id": "features", "keys": ["array"]}]
        },
        "output_definition": {
            "bindings": [{"type": "array", "source_id": "targets", "keys": ["array"]}]
        },
        "metadata_definition": {"bindings": []},
        "relationships": [],
        "sequence_definition": None,
        "subset_definition": {"type": "all"},
    }

    created = client.post("/api/datasets", json=payload)
    assert created.status_code == 201
    assert created.json()["sample_count"] == 4
    assert created.json()["inspection"]["sources"][0]["arrays"][0]["shape"] == [4, 3]

###############################################################################
def test_npz_inspection_exposes_keys_without_persisting_array_values(
    client: TestClient, tmp_path: Path
) -> None:
    source = tmp_path / "bundle.npz"
    np.savez(source, features=np.ones((3, 2), dtype=np.float32), targets=np.arange(3, dtype=np.int64))
    payload: dict[str, object] = {
        "name": "NPZ inputs",
        "dataset_type": "array",
        "sources": [{"type": "array", "id": "bundle", "path": str(source), "format": "npz"}],
        "input_definition": {"bindings": [{"type": "array", "source_id": "bundle", "keys": ["features"]}]},
        "output_definition": {"bindings": [{"type": "array", "source_id": "bundle", "keys": ["targets"]}]},
        "metadata_definition": {"bindings": []},
        "relationships": [],
        "sequence_definition": None,
        "subset_definition": {"type": "all"},
    }

    created = client.post("/api/datasets", json=payload)

    assert created.status_code == 201
    assert created.json()["inspection"]["sources"][0]["arrays"] == [
        {"key": "features", "shape": [3, 2], "dtype": "float32"},
        {"key": "targets", "shape": [3], "dtype": "int64"},
    ]

###############################################################################
def test_parquet_inspection_uses_schema_metadata_and_supports_input_only(
    client: TestClient, tmp_path: Path
) -> None:
    source = tmp_path / "events.parquet"
    parquet.write_table(pa.table({"message": ["a", "b"], "score": [1.0, 2.0]}), source)
    payload = _tabular_payload(source)
    payload["name"] = "Input-only parquet"
    payload["sources"] = [{**_table_source(source), "format": "parquet", "delimiter": None}]
    payload["input_definition"] = {"bindings": [{"type": "columns", "source_id": "table", "columns": ["message", "score"]}]}
    payload["output_definition"] = {"bindings": []}

    created = client.post("/api/datasets", json=payload)

    assert created.status_code == 201, created.text
    assert created.json()["format"] == "PARQUET"
    assert created.json()["selected_sample_count"] == 2
    assert created.json()["output_definition"]["bindings"] == []

###############################################################################
def test_delimited_inspection_rejects_duplicate_headers_and_malformed_rows(
    client: TestClient, tmp_path: Path
) -> None:
    duplicate = tmp_path / "duplicate.csv"
    duplicate.write_text("value,value\n1,2\n", encoding="utf-8")
    malformed = tmp_path / "malformed.csv"
    malformed.write_text("value,other\n1\n", encoding="utf-8")

    duplicate_response = client.post("/api/datasets/inspect", json={"dataset_type": "tabular", "sources": [_table_source(duplicate)]})
    malformed_response = client.post("/api/datasets/inspect", json={"dataset_type": "tabular", "sources": [_table_source(malformed)]})

    assert duplicate_response.status_code == 422
    assert "duplicate" in duplicate_response.json()["detail"]["message"].lower()
    assert malformed_response.status_code == 422
    assert "column counts" in malformed_response.json()["detail"]["message"]

###############################################################################
def test_range_subset_is_zero_based_and_inclusive(
    client: TestClient, tmp_path: Path
) -> None:
    source = tmp_path / "patients.csv"
    source.write_text("age,weight,diagnosis\n42,71,positive\n50,80,negative\n61,90,positive\n", encoding="utf-8")

    created = client.post("/api/datasets", json=_tabular_payload(source, subset={"type": "range", "start": 1, "end": 2}))

    assert created.status_code == 201
    assert created.json()["selected_sample_count"] == 2

###############################################################################
def test_missing_source_remains_registered_and_delete_does_not_remove_source(
    client: TestClient, tmp_path: Path
) -> None:
    source = tmp_path / "patients.csv"
    source.write_text("age,weight,diagnosis\n42,71,positive\n", encoding="utf-8")
    created = client.post("/api/datasets", json=_tabular_payload(source)).json()
    dataset_id = created["id"]
    source.unlink()

    missing = client.get(f"/api/datasets/{dataset_id}")
    assert missing.status_code == 200
    assert missing.json()["availability"] == {
        "status": "unavailable",
        "unavailable_source_ids": ["table"],
    }

    deleted = client.delete(f"/api/datasets/{dataset_id}")
    assert deleted.status_code == 204
    assert client.get(f"/api/datasets/{dataset_id}").status_code == 404

###############################################################################
def test_delete_registration_leaves_external_source_untouched(
    client: TestClient, tmp_path: Path
) -> None:
    source = tmp_path / "keep.csv"
    source.write_text("feature,target\n1,yes\n", encoding="utf-8")
    payload = _tabular_payload(source)
    payload["input_definition"] = {"bindings": [{"type": "columns", "source_id": "table", "columns": ["feature"]}]}
    payload["output_definition"] = {"bindings": [{"type": "columns", "source_id": "table", "columns": ["target"]}]}

    created = client.post("/api/datasets", json=payload)
    assert created.status_code == 201
    assert client.delete(f"/api/datasets/{created.json()['id']}").status_code == 204
    assert source.read_text(encoding="utf-8") == "feature,target\n1,yes\n"

###############################################################################
def test_update_relink_refreshes_definition_and_inspection_snapshot(
    client: TestClient, tmp_path: Path
) -> None:
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    first.write_text("age,weight,diagnosis\n42,71,positive\n", encoding="utf-8")
    second.write_text("age,weight,diagnosis\n42,71,positive\n50,80,negative\n", encoding="utf-8")
    created = client.post("/api/datasets", json=_tabular_payload(first)).json()
    update_payload = _tabular_payload(second)
    update_payload["name"] = "Relinked outcomes"

    updated = client.put(f"/api/datasets/{created['id']}", json=update_payload)

    assert updated.status_code == 200
    assert updated.json()["name"] == "Relinked outcomes"
    assert updated.json()["sources"][0]["path"] == str(second.resolve())
    assert updated.json()["sample_count"] == 2

###############################################################################
def test_dataset_registration_survives_application_restart(
    tmp_path: Path,
    app_factory: Callable[..., FastAPI],
) -> None:
    database = tmp_path / "restart.db"
    source = tmp_path / "restart.csv"
    source.write_text("age,weight,diagnosis\n42,71,positive\n", encoding="utf-8")

    with TestClient(app_factory(database_path=database)) as first:
        created = first.post("/api/datasets", json=_tabular_payload(source))
        assert created.status_code == 201

    with TestClient(app_factory(database_path=database)) as second:
        listed = second.get("/api/datasets")
        assert listed.status_code == 200
        assert listed.json()["datasets"][0]["name"] == "Patient outcomes"

###############################################################################
def test_split_definition_is_typed_persisted_and_validated(
    client: TestClient, tmp_path: Path
) -> None:
    source = tmp_path / "split.csv"
    source.write_text("age,weight,diagnosis\n42,71,positive\n50,80,negative\n61,90,positive\n72,95,negative\n", encoding="utf-8")
    payload = _tabular_payload(source)
    payload["split_definition"] = {
        "strategy": "random",
        "train_fraction": 0.8,
        "validation_fraction": 0.1,
        "test_fraction": 0.1,
        "seed": 17,
    }

    created = client.post("/api/datasets", json=payload)

    assert created.status_code == 201
    assert created.json()["split_definition"] == payload["split_definition"]

    payload["split_definition"] = {
        "strategy": "random",
        "train_fraction": 0.7,
        "validation_fraction": 0.1,
        "test_fraction": 0.1,
        "seed": 17,
    }
    invalid = client.post("/api/datasets/validate", json=payload)

    assert invalid.status_code == 200
    assert invalid.json()["valid"] is False
    assert any(item["code"] == "split_fractions_not_one" for item in invalid.json()["validation"]["errors"])

###############################################################################
def test_revalidation_persists_invalid_schema_and_recovers(
    client: TestClient, tmp_path: Path
) -> None:
    source = tmp_path / "revalidate.csv"
    source.write_text("age,weight,diagnosis\n42,71,positive\n50,80,negative\n", encoding="utf-8")
    payload = _tabular_payload(source)
    dataset_id = client.post("/api/datasets", json=payload).json()["id"]

    source.write_text("age,diagnosis\n42,positive\n50,negative\n", encoding="utf-8")
    invalid = client.post(f"/api/datasets/{dataset_id}/validate")

    assert invalid.status_code == 200
    assert invalid.json()["validation"]["status"] == "invalid"
    assert any(item["code"] == "missing_columns" for item in invalid.json()["validation"]["errors"])
    assert invalid.json()["sample_count"] == 2

    source.write_text("age,weight,diagnosis\n42,71,positive\n50,80,negative\n61,90,positive\n", encoding="utf-8")
    repaired = client.post(f"/api/datasets/{dataset_id}/validate")

    assert repaired.status_code == 200
    assert repaired.json()["validation"]["status"] == "valid"
    assert repaired.json()["sample_count"] == 3
    assert repaired.json()["selected_sample_count"] == 3

###############################################################################
def test_revalidation_preserves_previous_inspection_when_source_is_unavailable(
    client: TestClient, tmp_path: Path
) -> None:
    source = tmp_path / "preserve.csv"
    source.write_text("feature\n1\n2\n", encoding="utf-8")
    payload = _tabular_payload(source)
    payload["input_definition"] = {"bindings": [{"type": "columns", "source_id": "table", "columns": ["feature"]}]}
    payload["output_definition"] = {"bindings": []}
    created = client.post("/api/datasets", json=payload).json()
    source.unlink()

    revalidated = client.post(f"/api/datasets/{created['id']}/validate")

    assert revalidated.status_code == 200
    document = revalidated.json()
    assert document["validation"]["status"] == "invalid"
    assert document["inspection"]["sample_count"] == 2
    assert document["sample_count"] == 2

###############################################################################
def test_legacy_definition_without_split_definition_remains_loadable(
    client: TestClient, tmp_path: Path
) -> None:
    source = tmp_path / "legacy.csv"
    source.write_text("feature\n1\n", encoding="utf-8")
    payload = _tabular_payload(source)
    payload["input_definition"] = {"bindings": [{"type": "columns", "source_id": "table", "columns": ["feature"]}]}
    payload["output_definition"] = {"bindings": []}
    created = client.post("/api/datasets", json=payload).json()
    database = tmp_path / "test.db"
    with sqlite3.connect(database) as connection:
        raw_definition = connection.execute("SELECT definition_json FROM datasets").fetchone()
        assert raw_definition is not None
        definition = json.loads(raw_definition[0])
        definition.pop("split_definition", None)
        connection.execute("UPDATE datasets SET definition_json = ?", (json.dumps(definition),))
        connection.commit()

    loaded = client.get(f"/api/datasets/{created['id']}")

    assert loaded.status_code == 200
    assert loaded.json()["split_definition"] is None
