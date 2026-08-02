# Ports and Connections

Last updated: 2026-07-30

Ports have stable IDs, display names, kinds, required flags, multiplicity, and optional rank expectations. Tensor output can connect to tensor input. Tensor output can also connect to a tensor-list input. Mask and scalar kinds remain distinct. A non-multiple input accepts one edge. Connection creation must reject missing ports, incompatible kinds, duplicate occupancy, self-loops, and cycles.
