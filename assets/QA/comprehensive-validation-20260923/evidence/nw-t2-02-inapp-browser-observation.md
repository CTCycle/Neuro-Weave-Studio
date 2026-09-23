# NW-T2-02 in-app Browser observation

Last updated: 2026-09-23

- Date: 2026-09-23
- Source revision: `66d04b3c788baf8ba2921a7f3c1cd4c8541e1e44`
- URL: `http://127.0.0.1:6425/builder/883d935e-b7c2-4df1-aada-0f9f2adf6bde`
- Disposable project: `NW-T2-02 in-app Browser evidence`

The in-app Browser rendered the saved graph with two input branches, two Dense
nodes, a merge node, and an output. The graph showed five persisted edges. After
selecting **Validate**, the accessibility tree exposed input width 8, branch
width 16, merge width 16, and output width 16. Graph Health showed **No
findings** and **Ready**, and the page announced “Model graph is valid.”

The separate Playwright screenshots in this campaign capture the valid merged
graph and a rejected cycle. The manual Browser fixture was deleted through the
project API (HTTP 204); a subsequent lookup returned HTTP 404. The final
reloaded Model Library showed 0/0 models after the browser workflows cleaned
their fixtures.
