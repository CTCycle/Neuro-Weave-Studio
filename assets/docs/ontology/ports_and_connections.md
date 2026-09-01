# Ports and Connections

Last updated: 2026-09-01

Ports have stable IDs, display names, kinds, required flags, multiplicity, optional rank expectations, and parameter-driven enabled conditions. Tensor output can connect to tensor input. Tensor output can also connect to a tensor-list input. Mask and scalar kinds remain distinct. A non-multiple input accepts one edge. Connection creation and server validation reject missing or disabled ports, incompatible kinds, duplicate occupancy, self-loops, and cycles. Value-producing output ports can be selected explicitly or inferred when they are unconsumed; port identity is retained for multi-output nodes.
