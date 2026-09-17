# Dataset Management Architecture

Last updated: 2026-09-17

NeuroWeave datasets are persisted registrations of external data. A registration owns a typed definition and compact inspection/validation metadata; it never owns, uploads, copies, serializes, or deletes the referenced files, folders, images, tables, or arrays.

## Boundary and flow

The dataset workflow is `Type -> Sources -> Input / output -> Samples -> Review and validation -> Register`. `DatasetWizardComponent` is a shared shell for create and edit/relink. It uses the existing `FilesystemBrowserComponent` and `/api/filesystem/browse` boundary, caches inspection only while the wizard is open, invalidates that result when a source or interpretation changes, and lets the registration endpoint re-inspect authoritatively before saving. The explicit `/api/datasets/validate` route remains available for consumers that need standalone validation.

The enabled organizations are tabular, images, text, time series, array/tensor, and composed/multimodal. Auto Detect is represented in the capability registry as a disabled future option and is not a backend type. The registry owns source layouts, formats, labels, and future workspace capabilities so type behavior is not scattered across templates.

## Typed definition

The backend Pydantic models in `app/server/domain/datasets.py` and matching TypeScript contracts in `app/client/src/app/domain/dataset.types.ts` use discriminated unions for sources and bindings. Sources are CSV/TSV/Parquet tables, image folders, plain-text files/folders, or NPY/NPZ arrays. Bindings assign table columns, image samples, folder labels, text samples, or array keys to input, output, metadata, or unused roles. A file-to-table relationship supports exact filename, stem, or relative-path matching. Time-series definitions optionally reserve timestamp and group columns. A subset is `all`, `first_n`, or a zero-based inclusive range.

At least one input is required, while output is optional for unsupervised/input-only data. Role assignment is mutually exclusive and relationship/timestamp/group columns are structural rather than model fields. Composed layouts intentionally cover the common image/text plus table case instead of introducing a generic graph editor.

## Inspection and validation

`DatasetInspectionService` resolves paths, reads delimited headers and bounded dtype samples, reads Parquet schema/row counts through PyArrow, reads NPY/NPZ headers without loading arrays, and enumerates image/text files without decoding content. Exact file-to-table mapping checks duplicate and unmatched identifiers and retains only bounded examples in validation findings. The service returns compact metadata and does not put sample values in API responses or SQLite.

`DatasetService.validate` re-inspects the source and checks source layout, source IDs, binding compatibility, selected columns/keys, structural fields, mapping consistency, sample counts, non-empty data, and subset bounds. `POST /api/datasets` and `PUT /api/datasets/{id}` perform this validation authoritatively before saving. A registered definition can later become unavailable; list/get computes live path availability without changing the stored snapshot.

## Persistence and API

The `datasets` table stores relational identity/count/status/timestamps plus bounded `definition_json`, `inspection_json`, and `validation_json`. The repository participates in the existing unit-of-work and never commits independently. The HTTP surface is:

- `POST /api/datasets/inspect`
- `POST /api/datasets/validate`
- `POST /api/datasets`
- `GET /api/datasets`
- `GET /api/datasets/{id}`
- `PUT /api/datasets/{id}`
- `DELETE /api/datasets/{id}`

Delete removes only the registration row and explicitly leaves user sources untouched.

## Frontend catalog and workspace

`AppResourcesService` owns an independent dataset summary resource with readiness, retry, and refresh signals. `/datasets` is a persisted catalog with search, availability filtering, compact cards, and explicit deletion confirmation. `/datasets/:datasetId` provides a route-scoped `DatasetWorkspaceState` and nested Overview, Structure, Processing, and Split routes. Overview and Structure render stored metadata and current availability; Processing and Split are future-only route foundations with no controls or persisted drafts.

Training remains a session-only prototype, but its selector now reads persisted dataset summaries. No training loader, split materialization, preprocessing pipeline, cache, DataLoader, or framework dataset conversion is introduced here.
