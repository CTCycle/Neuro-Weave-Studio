# NW-T3-05 in-app Browser observation

Date: 2026-09-28
Launcher: official Windows launcher; frontend `http://127.0.0.1:6425/`; backend `http://127.0.0.1:9750/`
Route: `/models` → `/builder`

The Codex in-app Browser was used before the Playwright slice. The rendered
Builder showed the `Node catalog` heading and the `Recurrent 5 nodes` category.
After expansion, the accessibility tree exposed enabled buttons for all five
live recurrent definitions:

- Bidirectional Recurrent Layer
- GRU
- LSTM
- Recurrent Stack
- Simple RNN

The screenshot visibly showed the NeuroWeave Model Builder shell, the live
catalog count of 149 available nodes, and the expanded recurrent category.
This is rendered/accessibility-tree observation only; no spoken screen-reader
result is claimed. The durable rendered screenshot from the authoritative
browser run is [recurrent catalog rendered](nw-t3-05-recurrent-native-final/nw-t3-05-recurrent-catalog-rendered.png).
