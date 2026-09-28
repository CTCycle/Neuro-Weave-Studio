# NW-T3-03 in-app Browser observation

Date: 2026-09-28
URL: http://127.0.0.1:6430/models, then /builder

The fresh in-app Browser session used the official launcher frontend. The Builder rendered the Node catalog with 149 available definitions and the expected category counts. Searching for Conv2D reduced the available list to the single Conv2D operation. Selecting it materialized the node in the graph and rendered the Inspector with the node title and parameter controls, including Filters and Kernel Size. The empty graph correctly showed blocked health findings for its missing input and output connections.

This observation is visual Builder evidence for catalog search, materialization, and Inspector rendering. The temporary draft was not saved; the tab was closed after inspection. Save/reload and runtime behavior are covered by the serial Playwright evidence linked from the NW-T3-03 report.
