# NW-T1-04 Browser diagnostics

Last updated: 2026-09-23

## Official application view

The Environment page was opened in the Codex in-app Browser at
`http://127.0.0.1:6425/environment` after the official launcher reported
frontend readiness and backend health. The rendered page showed PyTorch as
usable, the Native framework API selected, CPU/default device controls, and no
active workers. TensorFlow and JAX were left unchecked in this official local
runtime view. No runtime job was started.

## Controlled preference workflow

The serial Playwright run uses a controlled metadata fixture for preference
selection. Its captured post-reload screenshot shows TensorFlow selected, the
Native framework API selected, device `gpu:0`, memory policy
`tensorflow_limit_mb`, and the saved 2048 MB value. It also shows the PyTorch
controls and no active workers. This is selection/persistence evidence only;
the fixture does not establish runtime execution.

Three instrumented browser scenarios asserted an empty `pageerror` list:
active backend/API/device/policy restoration after reload, malformed nested
preference recovery, and storage read/write failure. The remaining cases
asserted the visible selection and reconciled storage state.

The final Playwright log records six passing serial tests. It includes Node's
`NO_COLOR`/`FORCE_COLOR` environment warning; it contains no Angular
`NG0956` tracking warning. The final run exited successfully and port 6426 was
free afterward.

Screenshot: [active runtime preference after reload](nw-t1-04-active-preference-reload.png).
Detailed output: [serial Playwright log](../logs/nw-t1-04-playwright.log).
