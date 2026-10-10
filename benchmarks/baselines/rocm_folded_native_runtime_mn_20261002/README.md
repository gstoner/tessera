# gfx1201 folded native runtime-M/N image identity

Owner ROCM-MXFP4-W4A8-1 / E2E-REAL-6-ROCM-MATMUL-CACHE.
Sync ROCM-FOLDED-RUNTIME-MN-2026-10-02.

Typed frontend Graph -> Schedule -> Tile -> ROCm Target -> verified native
image projection -> typed LDS views/fragments/full-K WMMA/folded scale ->
ROCDL/LLVM -> HSACO -> checked runtime ABI -> RX 9070 XT.

The native generator and image projection share one strict named Target
contract validator. Projection admits only the original verified static
shape/policy/ABI/schedule and an attribute-only kernel with audited host
scaffolding. It removes the shape-dependent Schedule hash only from image
identity; each checked descriptor retains its original Schedule hash, Tile
digest, authored Target digest and payload SHA256.

M/N become runtime image dimensions. K, tile size, whole/partial row and
column classes, raster group, CU/WGP mode, prefetch, row guard, epilogue and
every other physical/module attribute stay in image identity. Different
profiles or K values do not share a kernel. Launch descriptors still own
exact buffer shapes and launch geometry. Runtime checks the image policy,
fixed K, edge classes and grid/block before probing HIP.

The default frontend package uses runtime M/N. runtime_mn=False provides
a compiler-owned static native control through the same Graph/Schedule/Tile
route. It is not a Python shader route. The frozen HIP emitter is a separate
independent numerical/performance control.

## Exact-device proof

- Combined native admission/projection, folded package and wide-scale/ragged
  numerical gate: 84 passed (device-tests.txt).
- Three randomized M/N shapes and different payloads share one composite
  image, payload and entry symbol; each has its own Schedule and payload hash.
- K changes miss the image. Runtime and static-native packages match bitwise.
- Physical-key/frontend gates: 42 passed (physical-key-tests.txt), including
  independent raster, mode, row-guard, prefetch and epilogue key changes and
  refusal of gfx1151.
- Wrong grid/block or static dimensions are rejected before HIP: 3 passed
  (geometry-tests.txt); wrong policy/classes/K are covered in the combined gate.
- Existing shared native identity/cache families with hardware enabled:
  130 passed, 9 skipped (shared-identity-tests.txt). Skips belong to other
  owning-device envelopes; no gfx1151 execution parity is claimed.
- Host WSL frontend/payload/diagnostic/pass gates: 307 passed.

Overflow recovery is finite despite the overflowing combined fp32 scale;
underflow recovery is nonzero bf16 despite the vanishing combined scale.
Zero partials, ragged bounds and independent fp64/HIP comparisons remain proved.

## Production timing

M=256, K=5120. Seven alternating trials, three rotating resident copies.
All runtime/static/HIP outputs match bitwise and independently sampled
reference values before timing. Compiler-built device-clock marker windows
exceed 5 ms and agree with HIP events within 5%; rejected short windows are
recorded and each accepted window is normalized by its actual launch count.
Device windows include host dispatch gaps, not profiler phase attribution.
Public launch wall time includes transfers/allocation/module load and is
separate from frontend/package cost.

| N | Image state | Runtime package ms | Static package ms | Runtime device us/launch | Static device us/launch | Runtime/static | Runtime public ms |
|---|---|---|---|---|---|---|---|
| 4096 | cold | 323.53 | 241.92 | 79.26 | 76.85 | 1.0314 | 17.22 |
| 8192 | warm_cache | 89.59 | 277.37 | 150.61 | 150.97 | 0.9977 | 44.98 |
| 16384 | warm_cache | 123.14 | 303.04 | 286.18 | 288.49 | 0.9920 | 82.86 |

All three runtime-M/N rows use one HSACO. Only the first image compile is cold;
later shapes reuse it. Frontend lowering and payload binding remain per shape
and explain why warm package cost is not zero. Public transfer/module overhead
is not fixed by an image cache.

Paired runtime/static device ratios are 0.9920–1.0314. This bounds the observed
specialization trade-off; it is not a GPU speedup claim. Runtime uses 178 VGPRs,
static 177, both 25,600 bytes LDS and no spills. The separate HIP control uses
123 VGPRs; its remaining native gap and column-cost attribution stay open.

K still keys the image. Runtime-K extension, register-lifetime attribution,
exact per-K32 native migration, wider scale/layout coverage and checkpoint/model
quality remain open. gfx1151 lacks FP8 WMMA under RDNA3.5. Apple/NVIDIA/x86
physical schedules and execution capabilities are unchanged.

## Reproduce

Source scripts/_rocm_env.sh, expose the matching compiler/LLVM dependencies,
and run from the owning WSL checkout:

    python benchmarks/rocm/record_gfx1201_folded_native_package.py --tessera-opt "$TESSERA_OPT" --llvm-bin "$LLVM_BIN" --native-static-control --output benchmarks/baselines/rocm_folded_native_runtime_mn_20261002/gfx1201.json

Packet includes live device/architecture, dirty-source fingerprints (including
the shared contract header and projection pass), compiler hash, selected-symbol
ISA/resources, native image cache states, original/image ancestry and every
timing window. A later frontend-only bool-argument check does not relabel
these image measurements as rerun.

Final supplementary mixed M/N panel classes: [mixed-panel-tests.txt](mixed-panel-tests.txt), 2 passed. Final audit/registry gates: [final-gates.txt](final-gates.txt), 311 passed. This packet remains fixed-K revision evidence; runtime K is in the subsequent packet.
