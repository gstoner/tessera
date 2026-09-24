# MXFP4 image producer relabel: timed kernels unchanged

Every hand-written MXFP4 packager (exact/WMMA W4A8, folded, safe-epilogue,
TN4, packed-folded) and the Quark W4A4 probe used to record
`pipeline_name="tessera-lower-to-rocm"` on its native image. No MLIR pipeline
produced those binaries: the source is hand-emitted HIP compiled by hipcc. They
now record the registered non-MLIR producer `hand-emitted-hip`
(`native_artifact.NON_MLIR_IMAGE_PRODUCERS`). Any Graph/Tile/Target IR a
delegate was reached through stays bound by `target_ir_digest` and descriptor
provenance.

The change touched only that label, but four sealed gfx1201 timing packets bind
the generator files' SHA-256. Following the repo's rule for old timings, those
packets are **not edited**: their tests now pin the generator hashes they
recorded (which equal the parent revision's files). This packet proves on
Tajasarus's RX 9070 XT that **every kernel they timed is the same kernel at the
relabel**:

- **Production WMMA kernels** (`gfx1201_mxfp4_kstep_prefill_20260922`, and the
  WMMA rows of the other three): rebuilt twice with each row's recorded
  schedule; the whole HSACO payload is byte-identical to the recorded
  `image_sha256` (these builds are deterministic).
- **Folded, safe-epilogue, TN4 and all ten packed-folded variants**, for three
  prefill shapes (39 builds): these builds embed build-specific bytes, so two
  builds of one revision already differ as whole payloads. The comparison is
  the selected symbol's `instruction_stream_sha256`, the same field the
  packets record. Parent revision and relabel produce identical streams for
  all 39, and all 15 streams recorded across
  `gfx1201_mxfp4_prefill_sweep_20260923`, `gfx1201_mxfp4_a_offset32_20260923`
  and `gfx1201_mxfp4_prefill_experiments_20260923` are among them.

`tests/unit/test_rocm_mxfp4_producer_relabel_evidence.py` verifies the chain:
the packets' recorded generator hashes equal this proof's parent hashes; this
proof's relabel hashes equal today's generator files (so any later generator
edit needs a new proof); and every timed kernel recorded in the four packets is
among the rebuilt kernels.

Reproduce (Tajasarus, repo root, ROCm env sourced):

```bash
PYTHONPATH=python:. python benchmarks/baselines/gfx1201_mxfp4_producer_relabel_20260924/check_production_payloads.py
PYTHONPATH=python:. python benchmarks/baselines/gfx1201_mxfp4_producer_relabel_20260924/check_instruction_streams.py
```
