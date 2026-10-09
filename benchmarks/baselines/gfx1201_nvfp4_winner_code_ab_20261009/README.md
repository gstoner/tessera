# gfx1201 native NVFP4 candidate-search loop

Owner: ROCM-NVFP4-INGEST-1 / E2E-REAL-6.
Synchronization key: NVFP4-MAGNITUDE-SEARCH-20261009.

Exact RX 9070 XT (gfx1201) proof uses the queried device identity and UUID.
The compiler route remains Graph → Schedule → Tile → ROCm Target → GPU MLIR →
ROCDL/LLVM → HSACO → checked native HIP ownership. Python supplies frontend
bindings and an independent oracle; candidate search and production math remain
native MLIR.

## Engineering decision

1. Winner-code rematerialization selected the exponent before encoding the final
   codes. It passed all 34 initial device cases and paired bitwise producer
   comparisons, but converter baseline/candidate ratios were 0.853–0.991.
   It was rejected. Its exact emitter snapshot and raw packet are retained as
   diagnostic evidence; the snapshot is not a production compilation input.
2. Magnitude-domain candidate SSE removes sign reconstruction while retaining
   signed output codes. For a common sign, the reconstructed difference changes
   only sign; its square is unchanged. Ordered midpoint comparisons, exponent
   tie order and the eight-accumulator sum32 reduction stay unchanged.
   This is the retained production change.

## Numerical and compiler gates

- 36 owning-device leaf/bounded-program cases pass. These include every finite
  positive E4M3 scale pair (127×127) with all signed E2M1 source codes, two
  projection globals, midpoint-adjacent globals, extreme scales, signed zero,
  ragged dispatch, changing active rows, portable replay and native lifetimes.
- 351 native compiler/ABI/diagnostic/pass-registry cases pass; three owning-image
  tests skip in the CPU host gate.
- Every measured arm validates independent conversion/storage/final oracles.
  Baseline/candidate packed codes, exponent bytes and f64 statistics are bitwise
  identical. Final product maximum absolute error is zero in the paired packets.
- Ruff and patch-integrity checks pass. Generic scaled batching/transpose remain
  open; their existing closure assertions have not been weakened or relabeled.

## Paired timings

Both magnitude packets use the same baseline binary, candidate binary, operands,
runtime providers and same device. Seven alternating arm rounds capture 128
launches per window. Converter windows contain 128 nodes; combined program
windows contain 384 nodes and one host graph submission. Baseline/candidate
images and tool/source hashes are retained. The baseline binary SHA matches the
previous native reshape evidence (947111d6aa8e0d119761332d384995c32260b8692146a34590a78942bd5a8c6e).
The stale-source warning for this deliberately preserved baseline is expected;
it is not used as the current-source compiler.

| Weight N×K | Converter baseline/candidate, two runs | Combined baseline/candidate, two runs |
| --- | --- | --- |
| 32×64 | 1.0458–1.0465 | 1.0326–1.0415 |
| 80×256 | 1.0505–1.0555 | 1.0412–1.0426 |
| 64×1024 | 1.0465–1.0494 | 1.0368–1.0417 |
| 256×2048 | 1.0060–1.0088 | 1.0094–1.0113 |

Each weight envelope runs active M=17 and M=257. Converter work depends on N/K;
M varies the bounded consumer, not converter work. Activation scale values change
between rounds. Warm package end-to-end wall samples include checks, transfers,
native execution and readback and are recorded separately. They are not kernel
time; stage medians are not summed to infer whole-program latency.

These are measured gains for explicit NVFP4 ingest in the named envelopes.
The widest-envelope gain is small. There is no physical selector promotion,
original-BF16/model-quality claim, FP8/MXFP8/MXFP4 promotion, or sibling evidence.

## Reproduce

After an exact gfx1201 device/toolchain probe, bind matching HIP runtime providers,
PYTHONPATH, TESSERA_OPT/TESSERA_ROCM_OPT and TESSERA_GFX1201_DEVICE_PROOF=1:

```sh
python -m pytest -q tests/device/rocm/test_native_nvfp4_ingest_leaf.py tests/device/rocm/test_nvfp4_bounded_rows.py
python benchmarks/rocm/record_nvfp4_winner_code_ab.py \
  --baseline /path/to/preserved/tessera-opt --baseline-source /path/to/baseline.h \
  --candidate /path/to/current/tessera-opt --experiment magnitude_sse \
  --output /path/to/packet.json
```

Run timing after test/Graphify jobs finish. The recorder rejects overlapping jobs.
Preserve the baseline from commit 386fa12ed8821ddc007c718525f0f4eb067e3353
before building the candidate. Future runs must inspect new device/tool hashes;
historical numbers are not current-device proof.

Remaining: candidate-search parallelization/physical schedule attribution,
whole-model quality, general layouts/batching/transpose and the wider
W8A8/MXFP4 performance obligations. NVIDIA W1.1 and saved-LSE forward/backward
still require their own completion audits.
