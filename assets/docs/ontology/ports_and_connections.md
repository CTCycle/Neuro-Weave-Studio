# Ports and Connections

Last updated: 2026-09-01

Ports have stable IDs, display names, kinds, required flags, multiplicity, optional rank expectations, and parameter-driven enabled conditions. Tensor output can connect to tensor input. Tensor output can also connect to a tensor-list input. Mask and scalar kinds remain distinct. `accepts_multiple` is target-input cardinality: a non-multiple input accepts one edge, while an output is allowed to fan out to every semantically compatible downstream input. Connection creation and server validation reject missing or disabled ports, incompatible kinds, duplicate occupancy, self-loops, and cycles. The only graph-level topology rule is acyclicity; port kind, enabled-state, duplicate, and declared input-cardinality rules remain semantic constraints. Value-producing output ports can be selected explicitly or inferred when they are unconsumed; port identity is retained for multi-output nodes.
