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
