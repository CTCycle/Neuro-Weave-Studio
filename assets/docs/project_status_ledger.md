# Project Status and Validation Record

Last updated: 2026-10-19

This ledger records the latest meaningful validation by product area and the boundaries of each claim. Architecture and ontology documents remain the source of behavior contracts; this file does not replace them.

The current 47-gate campaign snapshot is 42 PASS, 1 PARTIAL, and 4 UNTESTED. These are bounded gate results, not an overall product-readiness score.

## Status meanings

- VALIDATED: the named scope passed at the stated revision and date.
- PARTIAL: some tested behavior passed; material scenarios remain open.
- UNVALIDATED: implementation exists, but evidence is not sufficient for the named scope.
- NOT_IMPLEMENTED: the capability is outside the current implementation.

## Current evidence

The latest broad application-behavior evidence is from source revision bcca2e63e05d692cd64031c1318518dbc7cbfa17 on develop, dated 2026-10-18. The receipt revision d90e4bc adds test verification and browser evidence; the report states that application implementation files were unchanged. The focused 2026-10-19 launcher slice validated the working tree based on 48c6cdc51c543011c9def17e281d4463eb21db40; its launcher and test file hashes are recorded in the dated QA receipt.

| Area | Status and evidence | Remaining boundary | Contract |
| --- | --- | --- | --- |
| Builder and node catalog | VALIDATED. The live catalog contained 149 definitions with fingerprint acf538fb04f50552e1577299aa3f0cc09511d4e7d0937f73aa8418125a3a9ef8. The Tier 3 campaign recorded 20 bounded PASS slices on source revision 194c5134f2c9ebbe30d39f92c9e79a7673ff2eb1 (2026-09-30); the 2026-10-18 export run reconfirmed the live catalog on bcca2e63e05d692cd64031c1318518dbc7cbfa17. | Coverage is limited to recorded scenarios and profiles; exhaustive parameter, rank, and edge combinations are not claimed. | [Manifest specification](nodes/manifest_specification.md), [node taxonomy](ontology/node_taxonomy.md), [validation rules](ontology/validation_rules.md) |
| Execution profiles and compilers | VALIDATED. The 2026-10-18 export run exercised all six exact profiles on CPU (source revision bcca2e63e05d692cd64031c1318518dbc7cbfa17). Live functional and compiler gates also passed on 2026-10-01 in the working tree based on 68b3a0698935385073b11ccb2052a5199e825f43; unavailable compilers failed closed in ordinary launch. | GPU execution/export is not covered by the 2026-10-18 matrix. Compiler evidence remains toolchain-specific. | [Model execution profile](architecture/model_execution_profile.md), [framework compatibility](ontology/framework_compatibility.md) |
| Export artifacts and downloads | VALIDATED. On 2026-10-18, all 22 currently advertised profile/format pairs passed manifest, archive-member, hash, structural, reload, and inference checks in a 32-test selection with no skips. The in-app browser downloaded and reimported a project bundle that persisted with its graph and profile. | This does not cover every possible node/parameter/rank combination or GPU export. | [Export pipeline](architecture/export_pipeline.md) |
| Project and dataset persistence | VALIDATED. Project revision workflows passed on source revision d42bda2330581461fff2e52663780d0ea3fa5e53 (2026-09-23). Dataset metadata workflows passed on source revision 0ab274d51ad511cbda3a55c5a4599aade28d5404 (2026-09-25), covering inspection/revalidation, relink, and declarative Processing/Split persistence. | Dataset bytes remain external. Split settings do not create partitions or training data. | [Persistence](architecture/persistence.md), [serialization and versioning](ontology/serialization_and_versioning.md), [dataset management](architecture/dataset_management.md) |
| Startup, concurrency, and recovery | PARTIAL. NW-T0-02 remains PASS: ordinary launcher readiness and cleanup passed on 2026-10-19 under Windows PowerShell 5.1 and PowerShell 7. NW-T5-01 remains PARTIAL: the current launcher passed the occupied-port approval/rejection matrix, an eight-case frontend build-state/recovery matrix, and the focused launcher contract suite (14 passed) on the working tree based on 48c6cdc51c543011c9def17e281d4463eb21db40. PowerShell 5.1 metadata-probe and null-input failures were reproduced and fixed. See the [2026-10-19 launcher receipt](../QA/comprehensive-validation-20261019/launcher-startup-recovery.md). Prior evidence remains bounded: the single-server SQLite browser matrix passed 20/20 workflows at each worker count (60 total) on revision 9d1ab947a435d60ef2a957263f12a57b0f39f7e5; export conversion, atomic-publication, and cancellation failure injections preserved the previous artifact on bcca2e6. | NW-T5-01 still awaits natural runtime import/crash recovery and live external-service outage recovery. NW-T4-04, NW-T5-03, NW-T5-04, and NW-T5-05 remain UNTESTED. Injected failures do not prove recovery from a natural crash. | [Startup](runtime/startup.md), [validation strategy](development/validation_strategy.md), [export pipeline](architecture/export_pipeline.md) |
| Repository quality gates | VALIDATED at revision 194c5134f2c9ebbe30d39f92c9e79a7673ff2eb1 on 2026-09-30: strict Pyright, Ruff, backend suite, 105 frontend unit tests, lint, production build, launcher readiness, and deep-link/refresh checks passed. | The 2026-10-18 run was focused on export/recovery; it did not repeat the full repository suite. | [Testing strategy](development/testing_strategy.md), [CI workflow](../../.github/workflows/ci.yml) |

## Product and validation boundaries

- Training execution, optimizer steps, weight updates, checkpoints, and persistent training jobs are NOT_IMPLEMENTED. The Training page remains a session-only draft.
- Dataset split materialization and preprocessing are NOT_IMPLEMENTED. Dataset definitions and source metadata do not contain external data bytes.
- Custom public repository entries are references, not executable code. Trusted plugins and the live external-reference outage matrix remain outside the validated scope.
- Responsive checks and baseline accessibility assertions do not establish WCAG conformance or screen-reader acceptance. Manual assistive-technology coverage and comparable performance timings remain UNVALIDATED.
- The 2026-10-18 export matrix used CPU. CUDA availability was observed, but GPU export was not tested.

## Maintenance

- Keep one concise status row per durable product area; record the source revision, evidence date, scope, and material limitation.
- Put behavior rules in their linked architecture or ontology document and regression expectations in executable tests.
- Keep detailed run output under ignored assets/QA. Remove redundant logs, screenshots, manifests, and duplicate ledgers after their durable findings are summarized here.
- Update status only when new evidence changes the named scope; never widen a claim from metadata, static inspection, or one profile to an untested runtime or workflow.
