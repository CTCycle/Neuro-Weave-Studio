# Startup optimization validation

Date: 2026-08-31

## Measured findings

- Preview request to `/` crashed the pre-fix server with `EISDIR` in about 20 ms; the launcher then waited for the dead frontend until its 60-second deadline.
- Six live computational backend probes took about 26.3 s sequentially.
- Matching discovery snapshot validation in a fresh process took about 1.4 s.
- Python application imports took 1.03 s; isolated database initialization took 24 ms; node catalog initialization took 15 ms.
- Angular warm navigation through a working URL rendered the catalog heading in 96 ms and accepted search input.

## Launcher timing samples

Official `start_on_windows.ps1 -Action Launch -NoBrowser` runs used isolated discovery roots and stopped processes between samples. Optional runtime discovery was not run during launch.

| sample set | frontend ready median / max | backend ready median / max | total median / max |
| --- | ---: | ---: | ---: |
| discovery-cache-miss (3) | 605 / 609 ms | 1212 / 1220 ms | 3127 / 3180 ms |
| snapshot-backed (3) | 604 / 620 ms | 1208 / 1222 ms | 3190 / 3198 ms |

The frontend was ready before the backend health wait, and the browser handoff was performed immediately after frontend readiness.

## Runtime discovery

An isolated backend run returned `POST /api/execution/discovery` with HTTP 202 in 26 ms and completed the requested JAX/Keras probe in about 3.3 s. The probe ran outside startup and published a live result.

## Frontend and server validation

- Production build passed. Initial transfer was 95.77 kB; feature routes remained lazy (builder 20.00 kB, compilation 6.73 kB, export 5.26 kB, environment 4.58 kB transfer).
- Angular tests passed (3 files, 9 tests); lint passed.
- Development server compiled in about 3.96 s and served `/`, deep links, proxy health, metadata, and 404 missing assets.
- Preview served `/` and `/models`, returned 404 for a missing script, returned 400 for malformed URL encoding, and survived each request.
- Browser warm, cache-busting navigation samples rendered meaningful catalog UI in 82 ms median (95 ms max) and accepted input in 104 ms median (118 ms max).

Browser-tool cache-disabled navigation was unavailable, so that specific gate remains unverified. The visual screenshot was inspected in the in-app Browser but could not be persisted here because the existing `assets/QA` ACL rejected new screenshot files.

## Backend tests

The targeted final pytest set passed with 26 tests and 2 skipped. Pyright reported zero errors for changed backend files and Ruff passed. A broader suite was also run with protected cache directories ignored; its initial failures were old tests that assumed synchronous discovery and were updated to use the explicit preflight contract.

## Follow-up qualification

- The in-app Browser capability inventory exposes visibility and viewport controls, but no cache-disable or network-interception control. Three fresh query-busting `/models` navigations rendered the `Saved models` heading in 83–87 ms (goto 52–55 ms), with no browser console errors. These are cache-busting measurements, not a DevTools **Disable cache** measurement; that exact gate remains unavailable on this Browser surface.
- A disposable source snapshot was used for a Standard `start_on_windows.ps1 -Action Install -Backends none` run. The observed shell wall time was approximately 2m44s: portable Python 4.95 s, backend package installation 26.44 s, npm dependency installation 39 s, and Angular production build 68.474 s. The run downloaded and installed pinned uv/Python/Node runtimes and base dependencies without modifying this checkout. Optional backend download timings were not included.
- A process-cold launch after terminating the prior launcher-owned process trees reached frontend readiness in 1.17 s, backend readiness in 2.33 s, and reported `launcher.total` in 5.00 s. This is a stopped-process measurement, not an OS-reboot measurement; no reboot was performed.
- Fresh local checks confirmed `nvidia-smi` access to an NVIDIA GeForce RTX 3060 Laptop GPU with driver 610.88. Isolated PyTorch Keras and native probes both reported CUDA `cuda:0` with 6,441,926,656 bytes of reported device memory; TensorFlow and JAX probes reported their supported CPU paths. This is a claim about the exercised host/runtime only, not general hardware availability.
- Pytest cache and temporary directories now use `assets/QA`, generated `app/tests/cache` directories are excluded from collection, the Starlette test client uses `httpx2`, and the preview regression test closes its HTTP error stream. The full backend suite passed with `-W default` and emitted no warnings after these changes; the two prior Keras input-structure warnings were also removed by using named model inputs in the tests.
