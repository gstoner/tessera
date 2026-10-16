# gfx1201 bounded source-preserving NVFP4 candidate

Owner: ROCM-NVFP4-INGEST-1 / FRONTEND-IR-MEDIUM-1 / E2E-REAL-6.
Sync: NVFP4-SOURCE-BOUNDED-20261016.

Native NVFP4 v4 retains K16 source-preserving expansion and the exact FP8
consumer while giving activation, activation-scale and output buffers fixed
row capacity. Native MLIR owns Graph partition, Schedule/Tile and arbitrary-M
Target projection. The C++ owner validates active row spans and rebinds M,
grid and flattened memref extents without reallocating the capacity buffers.
Python marshals that ABI.

## Evidence

- device-ownership.txt: 59 passing bounded/static source, public replay and
  native partition cases on RX 9070 XT/gfx1201. Fifteen bounded cases cover
  FP32/BF16, reordered roots, full and activation-only updates, stable device
  allocation, retained outputs, stale generation and eight malformed row frames.
- fresh-process-replay.txt: the strengthened public test also passes fresh
  process replay at M1/64/19 with compiler paths disabled and numerical helpers
  forbidden. This test was extended after the 59-case run.
- device-regressions.txt: 226 passing prior bounded NVFP4, FP8 W8A8, MXFP8,
  exact MXFP4 and generic scaled-owner cases.
- host-regressions.txt: 544 passed, 22 skipped; the skips require hardware
  unavailable on the NVIDIA compiler host. They are not device proof.
- audit.txt: 11 passed.
- gfx1201-timing.json: nine synthetic rows, seven windows per timing scope,
  every window at least 10 ms. Independent nibble decoding and FP64 reference
  comparison precede timing and repeat after each window. Each bounded package
  retains image identity across its active shapes.

The manifest records candidate source hashes, tool donor identities, runtime
identity and receipt hashes. donor_source_and_compiler is the initial transfer
snapshot; source_sha256 records the later validation test revision. Compiler
and runtime implementation did not change between those measurements.

## Timing interpretation

Grouped producer/consumer graph-dispatch medians are approximately 3-4 us /
6-12 us. Ordinary program event medians are 9.81-14.43 us and can include host
enqueue gaps. Completed prepared invoke/read medians are 0.219-0.416 ms and
exclude preparation and input updates. These are separate timing scopes;
none is a pure-kernel latency or public-JIT speedup measurement.
Inputs are small synthetic dyadic controls, not original model quality proof.

## Reproduction

On the owning gfx1201 WSL host, from the matching repository checkout, source
~/scratch/nvfp4-source-bounded-20261016/env-candidate.sh. Run:

    python -m pytest -q tests/device/rocm/test_nvfp4_source_bounded_rows.py
    python benchmarks/baselines/nvfp4_source_bounded_20261016/record-bounded-source.py

The recorder companion removes only an extra blank line at EOF; manifest.json
retains the exact measured source in measured_recorder_utf8 and records both
hashes. The recorder companion intentionally retains the original scratch output
directory and imports the independently decoded test fixtures. Its writes go
under ~/scratch/nvfp4-source-bounded-20261016; preserve the manifest-bound input
code, provider and compiler configuration when comparing reruns.

## Open gates

Generated-document and broader PR validation remain pending. Legacy schema
meanings are unchanged. No general shape/layout/AD or model-quality closure,
performance promotion, gfx1151 FP8 execution or sibling-device proof is claimed.
