# ROCm attention ancestry, resident LSE and loading experiment

Owner: ROCM-2 / E2E-REAL-6 / EVIDENCE-PACKET-1.
Sync: `GFX1201-ATTENTION-PAIR-2026-09-13`.
Baseline `ca0e0c7263a443ea62974dffb7a33e6f786cff66` plus uncommitted sources
and compiler hashes in `source-identity.json`. Earlier commissioning packets
retain their original evidence and source identities.

## Implemented boundary

ROCm scheduled attention forward/backward must replay Schedule→Tile and
project descriptor fields before compilation. Forward f16/bf16 inputs produce
f32 output on ROCm; the shared verifier now models that ABI explicitly.
Matmul already has native replay/projection. No Graph constructor was deleted,
and general gfx1201 matrix/attention package guards are not widened here.

The gfx1201 native forward producer now uses eight-element RDNA4 operands and
its matching accumulator map. Device forward output and finalized LSE remain
allocated and are passed directly to saved-LSE backward. This is native-kernel
composition, **not** automatic public AD, a reusable resident tape API,
reader-aware lifetime management or asynchronous retirement.

## Validation

- `gfx1201-paired.txt`: 30 cases across f16/bf16, causal/ragged shapes, head
  dimensions 16/64. Three modes: oracle O + recomputed LSE; device O + saved
  LSE; device O + saved LSE with optional half-fragment forward loads.
  Device-fed modes do not upload oracle O or LSE. O, LSE and dQ/dK/dV are checked.
- `gfx1151-paired.txt`: 20 baseline/resident cases pass independently on
  Princess-Luna; ten gfx1201-only half-load experiments skip.
- `contracts.txt`: 32 passed, one skipped, 16 owning-device tests deselected.
  Includes native valid-parent projection and altered scale/workgroup/Tile
  refusal. The excluded x86 execution requires its shared image, unavailable
  in Tajasarus's ROCm-only build; no x86 execution claim.
- `lint.txt`: ruff and zero-error mypy ratchet pass.

Tajasarus uses assertions-enabled LLVM/MLIR 23.1.1, ROCm 10.0, RX 9070 XT
under Ubuntu 26.04 WSL2. Luna uses the isolated candidate compiler at
`/tmp/tessera-opt-gfx1201-wave`, with its layout-algebra loader path explicitly
set. Its main source checkout is not replaced by this experiment.

## ISA-guided experiment: not promoted

`half_fragment_loads=true` is an opt-in gfx1201 directive attribute. It
predicates the operand loads by 16-lane half and reconverges before matrix
instructions/barriers. It does not emit the full-EXEC-only RDNA4 transpose
load. Ragged load predicates remain in both paths.

`loading/comparison.json` binds each D=64 f16 forward kernel to source, compiler,
image and disassembly hashes. Counts are **static emitted instructions**, not
executed instruction counts or a profiler trace.

| Variant | VGPR | SGPR | LDS bytes | Spills | WMMA instructions |
|---|---:|---:|---:|---:|---:|
| Baseline | 119 | 48 | 5,312 | 0 | 8 |
| Half loads | 148 | 46 | 5,312 | 0 | 8 |

The higher register requirement argues against assuming an occupancy win.
Keep the default unchanged. `profiler-blocker.txt` records rocprofv3 failing
PC-sampling capability enumeration because WSL has no `/dev/kfd`. No runtime
kernel/counter attribution, overlap, timing or performance promotion is claimed.

Reproduce after sourcing the owning host environment:

```sh
TESSERA_ROCM_CHIP=gfx1201 python -m pytest tests/unit/test_rocm_flash_attn_bwd_compiled.py -k matches_numpy -q
python -m benchmarks.rocm.record_attention_loading --directory /tmp/attention-loading
```

Next gates: compiler-owned gfx1201 matrix/attention package admission; automatic
paired public AD with explicit f32-output/low-precision-cotangent conversion;
resident tape ownership; general masks; clean native-Linux profiler attribution
before tuning or promotion. ISA reference: the RDNA4 archive §7.12.2 and §11.6.2,
whose source matches Princess-Luna's AMD_GPU_ISA_DOCS manual byte-for-byte.
