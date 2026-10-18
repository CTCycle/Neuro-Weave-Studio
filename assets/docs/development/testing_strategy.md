# Testing Strategy

Last updated: 2026-10-18

The automated suite protects domain invariants, execution safety, persistence, and important user workflows. Keep tests where failures could corrupt a graph, select an unintended implementation, publish an invalid artifact, or block a core workflow. The [validation strategy](validation_strategy.md) defines evidence boundaries; the [status ledger](../project_status_ledger.md) records the latest scoped results.

## Main checks

Run the checks relevant to the changed code.

~~~powershell
cd app/server
uv run ruff check . ../tests ../scripts
uv run pyright
uv run pytest ../tests

cd ../client
npm run lint
npm run test
npm run build
npm run e2e
~~~

Focused backend tests live in app/tests; frontend unit tests live beside source files; Playwright end-to-end specs live in app/client/e2e. The CI workflow in .github/workflows/ci.yml defines the automated repository checks.

## Coverage boundaries

- Optional runtime tests depend on installed frameworks. The native-parity CI matrix selects one required backend per job and fails if that selected backend cannot be imported; ordinary local runs may skip unavailable optional runtimes.
- Runtime, compiler, and export claims are profile-specific. A helper or fake worker verifies only the contract it exercises; it does not substitute for live framework execution.
- External public-repository round trips are opt-in because service availability is not a deterministic CI dependency. Hermetic tests cover resolver failures and malformed references.
- Playwright file-input coverage does not certify every operating-system file chooser. Accessibility assertions are a baseline, not WCAG conformance or screen-reader acceptance.
- Functional Validation and Tensor Inspection use synthetic inputs and initialized weights. They do not test training, learned weights, or prediction serving.

Keep temporary test output under the ignored assets/QA directory. Preserve only durable findings in the owning concept documentation and status ledger.
