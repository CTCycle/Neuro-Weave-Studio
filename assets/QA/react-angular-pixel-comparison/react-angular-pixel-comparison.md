# React to Angular migration: rendered UI comparison

Date: 2026-08-31

Repository: `CTCycle/Neuro-Weave-Studio`

Baseline: `develop` at `a1f2e9878e8c7c731f455e32bec52cc9ae472a43`

Migration branch: `refactor/react-to-angular-migration`
Migration head: `c26c6ee` (`fix: align graph ports and connection hit areas`)

## Method

The comparison used the Codex in-app Browser against the local Angular development server and the local FastAPI backend. The backend fixture was project `b3245198-fb84-44cd-8dce-909e6628184f` (`QA Native PyTorch Binary`, 4 nodes, 3 edges, revision 5).

The matrix covered the model catalog and builder at 1023x768, 1024x768, 1199x800, 1200x800, 1440x900, 560x800, and 900x700. The workflow routes were captured at 1440x900: overview, compilation, export, inference, environment, and metrics.

Each result compares the RGB pixels of the Angular PNG against the matching develop capture. `exact` is the percentage of pixels with no channel difference; `>10` is the percentage of pixels with any channel difference above 10; `mean` is the mean absolute channel difference; `p95` is the 95th percentile channel difference.

The matrix develop `.jpg` files are the dimensionally correct baselines. Several non-1440 matrix `.png` files are legacy 1280x720 files despite their names, so they were not used for numerical comparison. The route `.png` files are dimensionally correct.

## Pixel results

### Model catalog

| Viewport | exact | >10 | mean | p95 |
| --- | ---: | ---: | ---: | ---: |
| 1023x768 | 99.67% | 0.06% | 0.02 | 0 |
| 1024x768 | 99.64% | 0.05% | 0.02 | 0 |
| 1199x800 | 99.75% | 0.05% | 0.02 | 0 |
| 1200x800 | 99.74% | 0.04% | 0.02 | 0 |
| 1440x900 | 99.86% | 0.04% | 0.01 | 0 |
| 560x800 | 99.54% | 0.14% | 0.04 | 0 |
| 900x700 | 99.71% | 0.06% | 0.02 | 0 |

### Builder

| Viewport | exact | >10 | mean | p95 |
| --- | ---: | ---: | ---: | ---: |
| 1023x768 | 60.48% | 7.04% | 3.79 | 11 |
| 1024x768 | 60.78% | 7.22% | 3.80 | 11 |
| 1199x800 | 65.37% | 6.86% | 3.49 | 11 |
| 1200x800 | 65.38% | 6.85% | 3.49 | 10 |
| 1440x900 | 67.03% | 6.42% | 3.20 | 10 |
| 560x800 | 62.44% | 5.08% | 2.31 | 8 |
| 900x700 | 71.65% | 7.82% | 2.60 | 11 |

The builder’s remaining pixel delta is concentrated in the Foblex canvas renderer and its connection/minimap rasterization. The result is a rendered-parity comparison, not a claim of byte-identical pixels.

### Workflow routes at 1440x900

| Route | exact | >10 | mean | p95 |
| --- | ---: | ---: | ---: | ---: |
| overview | 79.25% | 5.37% | 1.92 | 9 |
| compilation | 75.05% | 12.78% | 2.89 | 12 |
| export | 86.26% | 3.74% | 1.63 | 6 |
| inference | 67.56% | 19.28% | 4.55 | 12 |
| environment | 99.69% | 0.06% | 0.03 | 0 |
| metrics | 71.80% | 5.63% | 3.24 | 10 |

## Browser spot checks

The following captures are stored beside this report:

- `angular-final-builder-1440x900.png`: baseline builder render.
- `angular-final-builder-selected-node-1440x900.png`: selecting `Input Features` updates the inspector.
- `angular-final-builder-validation-1440x900.png`: validation panel reports `No findings` and `Ready` for the fixture.
- `angular-final-builder-focus-1440x900.png`: focus mode is visible while catalog and inspector remain available in the DOM.
- `angular-final-builder-minimap-hidden-1440x900.png`: minimap hides and the `Show minimap` action appears.
- `angular-final-builder-dirty-1440x900.png`: model name draft state is rendered.
- `angular-final-builder-catalog-drawer-1022x768.png`: catalog drawer opens at narrow width.
- `angular-final-builder-inspector-drawer-1022x768.png`: inspector drawer opens at narrow width.
- `angular-final-builder-selected-edge-1440x900.png`: native Foblex connection hit-testing visibly selects the edge.
- `angular-final-{overview,compilation,export,inference,environment,metrics}-1440x900.png`: workflow route captures.

The in-app browser reports a 1024px inner width for the 1023px request, and a one-pixel larger inner height for the 1199x800, 1200x800, and 560x800 requests. The saved image files retain the requested dimensions shown in their names.

## Validation gates

- `npm run lint -- --no-cache`: passed.
- `npm test -- --run`: passed, 3 files and 8 tests.
- Angular production build with an isolated output path: passed; output was removed after verification.
- `npm run e2e -- --workers=1`: passed, 4 tests in 47.9 seconds.
- Backend `ruff check app\server`: passed.
- Explicit backend test files with isolated QA cache/temp roots: 134 passed, 3 failed. The failures are untouched by this branch: two schema-v2 example fixtures conflict with the current schema-v3 contract, and the existing PyTorch LSTM fixture is rejected by its output-activation validation. Default pytest discovery also encounters access-denied protected cache directories under `app/tests/cache`, so the explicit-file rerun was used.
- `git diff --check develop...HEAD`: passed.
- Tracked client source contains zero `.tsx` files. No React, XYFlow, Zustand, or React icon runtime remains; Vite references are indirect Angular/Vitest lockfile dependencies, and migration references in docs are historical/architectural.

## Incremental publication

The branch was created without a worktree and pushed incrementally. Commits, oldest first:

`38ac6f6`, `22fa13e`, `babd3b5`, `1e0b63b`, `416c3fb`, `8cd3ad9`, `a9e022f`, `bd8d213`, `b46b843`, `56eb908`, `6d935c2`, `af2d02a`, `fb04fbe`, `c26c6ee`.

At report time, `HEAD` matches `origin/refactor/react-to-angular-migration` with zero commits ahead or behind. The unrelated pre-existing working-tree modification to `start_on_windows.ps1` was preserved and was not staged.

## Boundaries

This report records strong rendered parity for the catalog and most workflow surfaces, but not pixel-perfect identity for the graph canvas. The edge capture verifies Foblex’s native visual hit target; Angular signal-store edge deletion and inspector synchronization were not claimed as verified by that pointer-only check. Native TensorFlow/checkpoint capability claims were not made because those local optional runtimes were unavailable. The automated E2E suite covers four representative workflows rather than every scenario in the supplied audit.
