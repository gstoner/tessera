<!-- MERGE_START -->
# EBT v2.2 — Filled Pipelines, Backend Smoke, Roofline/Perfetto

**What’s new**
- **Pipelines**: concrete canonicalize/lower sequences (textual pipeline specs).
- **Backend smoke**: sample Tile→Target IR for NVIDIA (WGMMA), ROCm (MFMA), CPU (AVX2) with FileCheck tests.
- **Reporting**: a tiny roofline HTML and Perfetto JSON stub, plus a `--report` flag in the wrapper.
<!-- MERGE_END -->
