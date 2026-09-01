# Recurrent Layers

Last updated: 2026-09-01

Simple RNN, LSTM, GRU, bidirectional recurrent layers, and recurrent stacks require rank-three inputs shaped as batch, time, and features. `return_sequences` determines whether the time dimension is retained; `return_state` exposes additional hidden/cell output ports for LSTM/GRU and those ports participate in output selection and serialization. Keras mappings expose recurrent activations, initializers, regularizers, constraints, dropout/state controls, and runtime options. LSTM and GRU parameter counts use standard gate multipliers when input feature size is concrete. Native support is capability-limited and does not silently enable unsupported recurrent options.
