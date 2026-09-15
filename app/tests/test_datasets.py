from __future__ import annotations

import json
import sqlite3
from importlib import import_module
from pathlib import Path
from typing import Protocol, cast

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import create_app


class _PyArrowModule(Protocol):
    def table(self, data: object) -> object: ...


class _ParquetModule(Protocol):
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
    assert validation.json()["valid"] is False
    assert any(item["code"] == "missing_columns" for item in validation.json()["validation"]["errors"])

    created = client.post("/api/datasets", json=payload)
    assert created.status_code == 422
    assert created.json()["detail"]["code"] == "dataset_definition_invalid"


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
    assert {item["key"] for item in created.json()["inspection"]["sources"][0]["arrays"]} == {"features", "targets"}
    assert "1.0" not in json.dumps(created.json())


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
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "restart.db"
    source = tmp_path / "restart.csv"
    source.write_text("age,weight,diagnosis\n42,71,positive\n", encoding="utf-8")
    monkeypatch.setenv("DATABASE_SQLITE_PATH", str(database))
    monkeypatch.setenv("NEURO_WEAVE_EXPORT_ROOT", str(tmp_path / "exports"))
    monkeypatch.setenv("NEURO_WEAVE_ML_JOB_ROOT", str(tmp_path / "jobs"))
    monkeypatch.setenv("NEURO_WEAVE_RUNTIME_DISCOVERY_PATH", str(tmp_path / "environment.json"))

    with TestClient(create_app()) as first:
        created = first.post("/api/datasets", json=_tabular_payload(source))
        assert created.status_code == 201

    with TestClient(create_app()) as second:
        listed = second.get("/api/datasets")
        assert listed.status_code == 200
        assert listed.json()["datasets"][0]["name"] == "Patient outcomes"
