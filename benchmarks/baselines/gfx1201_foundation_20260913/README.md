# RX 9070 XT / gfx1201 foundation commissioning

Owner: ROCM-2, F0/F2; synchronization key `GFX1201-FOUNDATION-2026-09-13`.
Source baseline: merged `ca0e0c7263a443ea62974dffb7a33e6f786cff66` plus the
source hashes in this packet. These are uncommitted-increment measurements.

Tajasarus is an RX 9070 XT under Ubuntu 26.04 WSL2, ROCm 10.0/HIP 7.15.
It is not an AI PRO R9700 measurement and not native-Linux profiler proof.

- `assertions.json`: LLVM 23.1.1 reports ON and the invalid SmallVector operation
  aborts with SIGABRT. Tessera opt/ROCm opt/translate use those assertion-enabled
  libraries and matching no-RTTI compilation. The host companion C compiler and
  disassembler are the existing Ubuntu LLVM 23.1.1 binaries; neither is claimed
  to be an assertions-enabled clang build.
- `fragments.json`: compiler-emitted Tile → ROCDL → HSACO, six matrix dtypes,
  three shapes each, signed small-integer inputs and exact f32/i32 oracle.
  Ragged inputs are externally zero-padded: this does not prove runtime-bounded
  Tile loads or general K-loop GEMM. No timing is taken.
- `arithmetic.json`: scalar/vector add/sub/mul/div, 17 dtypes × two lane widths.
  FP8 arithmetic extends/rounds explicitly; passing it does not imply native FP8
  arithmetic instructions or packed/scaled matrix support.
- `runtime-tests.txt`: shipped HIPRTC f16/bf16 GEMM and forward attention.
  LDS/pipelined GEMM remains unavailable; the recorded skips do not count as
  device proof. Scheduled attention/VJP packages remain gfx1151-bound.

The matrix probe found an exact transpose in the RDNA4 accumulator store.
Correct row-strided stores require 25 VGPRs for f16/bf16 and 40 for INT4,
with zero spills in the compiler fixture. The old 24/36 ceilings measured the
incorrect transpose; resource ratchets are updated for correctness, not as a
performance promotion. gfx1200 receives the shared RDNA4 mapping correction
but has no owning-device proof here. Assertions additionally exposed missing
ROCDL/LLVM dependent-dialect declarations in Target-to-ROCDL, now fixed.

Reproduce on the owning host after sourcing `~/.config/tessera/env.sh`:

```bash
python scripts/probe_llvm_assertions.py --llvm-config "$LLVM_CONFIG" --output assertions.json
python -m benchmarks.rocm.record_gfx1201_fragments --output fragments.json
python -m benchmarks.record_dtype_arithmetic --backend rocm --chip gfx1201 \
  --compiler "$TESSERA_OPT" --llvm-bin "$TESSERA_LLVM_BIN" \
  --toolkit "$ROCM_PATH" --artifacts /tmp/gfx1201-arithmetic --output arithmetic.json
```

Remaining gates: compiler-owned multi-tile/K-loop admission; general scheduled
packaging, masks and saved-LSE/backward; packed/scaled/sparse matrix formats;
resource and kernel-clock attribution and independent-process promotion.
