# NW-T3-13/NW-T3-14 in-app Browser observation

Date: 2026-09-29

The Codex in-app Browser was pointed at the official frontend on
`http://127.0.0.1:6425/builder`. The live Builder rendered the catalog with 149
nodes and the Mathematical category. Searching for `math.equal` exposed the
real rendered button `Equal math.equal · elementwise`; adding it displayed the
elementwise node with tensor-list inputs and an output port. The draft was not
saved as a fixture. The authoritative rendered catalog screenshot and all
saved positive runtime screenshots are in this directory.
