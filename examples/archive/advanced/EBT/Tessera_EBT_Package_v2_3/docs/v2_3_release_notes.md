<!-- MERGE_START -->
# EBT v2.3 — Registered Pipelines, Grad-Path Pass, End-to-End Multi-Backend, CSV→Roofline

**Highlights**
- **PassPipelineRegistration** C++ scaffold (drop-in for your MLIR build).
- **tessera-ebt-select-grad-path** pass that swaps autodiff VJP with custom JVP on demand.
- **End-to-end** K×T samples that target NV/ROCm/CPU with FileCheck on backend ops.
- **CSV→HTML roofline** tool that ingests `reports/roofline.csv` and emits `reports/roofline.html`.
<!-- MERGE_END -->
