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
- final-drift-gates.txt: 401 passed across generated documentation, runtime ABI,
  audit lifecycle, diagnostics and pass metadata.
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
    python benchmarks/baselines/nvfp4_source_bounded_20261016/record_bounded_source.py

The recorder companion removes only an extra blank line at EOF; manifest.json
retains the exact measured source in measured_recorder_utf8 and records both
hashes. The recorder companion intentionally retains the original scratch output
directory and imports the independently decoded test fixtures. Its writes go
under ~/scratch/nvfp4-source-bounded-20261016; preserve the manifest-bound input
code, provider and compiler configuration when comparing reruns.

## Open gates

Final local generated-document and ABI drift gates pass; broader PR CI and
review remain pending. Legacy schema
meanings are unchanged. No general shape/layout/AD or model-quality closure,
performance promotion, gfx1151 FP8 execution or sibling-device proof is claimed.

## PR960 first-head unit repair

The first unit lane at 06dca840d has 21,957 passes, 9,974 skips, one failure and
18 fixture errors. pr960-unit-head06dca.log.xz preserves that red receipt.
The new companion used a hyphenated stem, which the recorder-discovery gate's
identifier tokenizer could not recognize. It is renamed record_bounded_source.py;
the empty unnamed-recorder ratchet is unchanged.

The host C++ readback fixture extracted the production read bodies but omitted
their new activeBytes helper and Program row fields. It now extracts the actual
production helper too. Four pinned/pageable, single/batched tests additionally
prove active-prefix reads and capacity-span refusal before enqueue. No native
runtime implementation or GPU arithmetic changes. ci-repair-red.txt.xz reproduces
the failures; ci-repair-green.txt records 23 passing tests.

## Original-checkpoint bounded source proof

original_checkpoint/gfx1201.json uses the same pinned Qwen3 gate/up NVFP4
tensors, BF16 sources and captured layer-0 activation input as the prior static
original-composed packet. Full source dictionaries and tensor hashes match.
N=24576, K=4096; one capacity-256 package and one owner execute active
M19/1/256/200/127/19 with unchanged images and capacity allocations.

All independent nibble-decoding/FP64 arithmetic bounds pass. Original public
JIT and portable replay also pass M1/19/200/256/1 with compiler subprocesses and
eager helpers forbidden. At M256, native relative RMS against the decoded FP8
projection is 2.568e-7; relative RMS against the original BF16 projection is
6.2286%. This remains projection-level proof, not full MLP/model acceptance.

Seven windows per active frame retain producer/consumer graph dispatch,
program events and completed activation-update/consumer/read scopes separately.
All windows are >=10ms; no unmatched-run speedup or default promotion follows.
Public/portable wall samples are single correctness-call characterizations,
not benchmark medians.

Reproduction companions: record_original_bounded.py and
cache_original_checkpoint.py under original_checkpoint/. The latter populates
one scratch range/projection cache. Validated NPZ projections are retained;
257,949,696 bytes of redundant raw tensor ranges were removed, with request
identities/digests retained in range-dedup-receipt.json. Model data is not
committed. Both companions retain the original owning scratch paths.

The source/tool/provider attestation binds the tested implementation to PR960's
06dca840d head. Subsequent CI repairs change only test infrastructure,
documentation and evidence. Git checkout refreshed source mtimes after the
matching tools were copied; the warning is retained and the source/binary
SHA-256 checks establish matching code without changing timestamps.

The first-head production compiler lane passes 1,265 cases with zero skips;
pr960-compiler-head06dca.log.xz retains that receipt. Repaired-head CI must
pass independently before merge.
