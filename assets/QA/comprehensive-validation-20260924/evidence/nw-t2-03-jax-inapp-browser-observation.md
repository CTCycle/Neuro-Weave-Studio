# In-app Browser observation — native JAX reload

Date: 2026-09-24

- Official preview: `http://127.0.0.1:6425`
- Test model: `90d7216d-c5f2-4044-9b77-a3a1b8a298e6`, saved revision 3
- Persisted profile: native JAX/Flax; runtime preference `native · jax · cpu`
- Functional validation reached `succeeded` on `cpu:0`: one input, one output, 672 parameters, 2230.8 ms.
- After a page reload, the accessibility tree still showed job state `succeeded` and the completed result summary for the same revision/profile.

The complete worker response is [nw-t2-03-jax-functional-validation-job.json](nw-t2-03-jax-functional-validation-job.json). The test model was created for this isolated validation database and is removed during campaign cleanup.
