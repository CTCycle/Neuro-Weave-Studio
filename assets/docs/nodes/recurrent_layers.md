# Recurrent Layers

Last updated: 2026-07-30

Simple RNN, LSTM, GRU, bidirectional recurrent layers, and recurrent stacks require rank-three inputs shaped as batch, time, and features. `return_sequences` determines whether the time dimension is retained. LSTM and GRU parameter counts use standard gate multipliers when input feature size is concrete.
