# NW-T3-02 in-app Browser observation

Date: 2026-09-28

The official launcher was open at `http://127.0.0.1:6428` while the bounded
workflow was inspected in the Codex in-app Browser.

- Runtime Environment: the PyTorch/Keras check reported one operational runtime;
  PyTorch `2.13.0+cu126` was usable, with CPU selected and the CUDA device
  visible.
- Model Builder: the live catalog reported 149 available definitions.
- GRU materialization: the Inspector exposed `Units`, `Return sequences`, and
  `Return state`; enabling the state option rendered the `Hidden state` output
  port.
- LSTM materialization: enabling the state option rendered both `Hidden state`
  and `Cell state` output ports.
- The final rendered runtime state is captured in
  [the recurrent-variants success screenshot](nw-t3-02-recurrent-variants-runtime-succeeded.png).

This is rendered UI evidence for the selected Keras/PyTorch lane. It does not
claim assistive-technology speech output or full profile/catalog coverage.
