# Graph Entities

Last updated: 2026-09-01

Node Definition is the immutable manifest contract. Node Instance is a placed and configured use of that definition and may be locked without becoming read-only. Port belongs to a definition. Edge connects one source output port to one target input port. Model Input and Model Output are semantic roles; output adapter nodes remain valid, while any non-input computational node may also provide an output binding. Explicit output bindings take precedence over inferred unconsumed terminal ports, and both node and port identity are preserved. Every edge endpoint must exist, node and edge IDs are unique, and the complete model dataflow must remain acyclic.
