# Backend Architecture

Last updated: 2026-08-21

`app/server/app.py` is the FastAPI application factory. Routers translate HTTP contracts, services own validation, import, export, and repository transactions, and the domain models define the graph, execution profile, runtime, and job contracts. Lifespan creates the catalog, repository context, and one application-scoped `RuntimeController`; the controller is shut down and joined before repositories are disposed.

`ModelExecutionProfile` is the backend authority for a project: `execution_backend` selects TensorFlow, PyTorch, or JAX and `keras_enabled` selects the Keras implementation versus the native implementation. The compiler and validator receive this pair explicitly, derive the manifest implementation key, and never recover a different backend from a worker preference or request-local fallback.

## Process-isolated runtime

The server process never imports Keras, TensorFlow, JAX, or PyTorch. `RuntimeController` discovers capabilities with short-lived probe workers and starts a new `models.ml_worker` interpreter for every smoke-inference or Keras-export job. The worker sets `KERAS_BACKEND` and allocator/device environment variables before importing Keras-dependent modules. This boundary avoids inherited accelerator state and unsafe framework reuse.

`WorkerConfiguration` is frozen and contains `api` (`keras` or `native`), the profile backend, normalized physical `device_id`, backend-specific memory policy, and graph-derived precision. Keras is operational when its selected backend probe succeeds. Native inference is deliberately represented as unavailable with `reason_code=not_implemented`; no native runtime fallback to Keras occurs.

The controller owns one lease per normalized physical device (`cpu`, `cuda:0`, `rocm:0`, `mps:0`, or `xpu:0`). Jobs on different leases may run concurrently. States are `starting`, `running`, `stop_requested`, `stopping`, `succeeded`, `terminated`, and `failed`. A lease is released only after the worker PID has been waited on and any termination escalation has completed.

Each job has a bounded workspace under `app/resources/runtime/ml-jobs` containing immutable input/configuration, atomic status/result publication, a cancellation marker, and bounded diagnostics. Graph input is removed after worker startup; terminal workspaces are removed. At most 100 terminal records remain in controller memory. Cancellation checks surround import, initialization, construction/loading, inference, save/reload, and publication. Cooperative cancellation receives five seconds, then the worker process tree is terminated and, after another five seconds, force-terminated.

Keras export writes a temporary `.keras` file in the job workspace. Reload/inference verification and a final project-revision check must pass before atomic replacement of the canonical export under `app/resources/runtime/exports`; failures and revision conflicts never overwrite a prior artifact.

## Runtime diagnostics and limits

Metadata is live probe output: API capability, installed/operational backend, discovered devices, supported memory policies, and normalized lease keys. Workers report PID, immutable configuration, lifecycle phase, exit code, cancellation/escalation state, restart reason, process RSS, and nullable framework/device metrics. Cleanup (`clear_session`, JAX cache clearing, PyTorch synchronization/cache cleanup, and garbage collection) is best effort and does not replace process teardown. PyTorch allocated versus reserved/cached memory and framework/device metrics are intentionally shown separately.

Training, checkpoints, datasets, and native-framework inference are outside this runtime boundary. Installing or removing optional packages still requires an application restart. Normal backend/device/policy changes only affect the next fresh worker; an operating-system restart is exceptional recovery for driver/runtime faults.
