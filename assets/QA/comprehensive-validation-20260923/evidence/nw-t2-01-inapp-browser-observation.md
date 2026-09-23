# NW-T2-01 in-app Browser observation

The official preview at `http://127.0.0.1:6425` served the built Builder against
the isolated backend on port `9849`. In the 4-node project fixture, the
in-app Browser rendered the Node catalog, canvas, model toolbar, and Inspector.

Selecting **Hidden layer** showed its `Units` field at 32. After changing it to
40, Undo restored 32 while leaving **Hidden layer** selected and its Inspector
open; Redo restored 40 with the same selection. Undo returned the fixture to
its saved value before the repair check.

Entering `1.5` showed **Invalid** and “Enter a valid units.” The field remained
editable while **Save model** and **Validate** were disabled. Replacing it with
24 produced a saved revision with the Inspector still open and the graph
showing the repaired width. The corresponding test screenshots and automated
browser diagnostics are linked from [NW-T2-01](../NW-T2-01.md).

This was a run-owned fixture in the isolated validation database. It was
deleted and verified absent before the database directory was removed.
