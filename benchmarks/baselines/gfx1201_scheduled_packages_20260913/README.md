# GFX1201 scheduled matrix and forward-attention packages

Owner: E2E-REAL-6 / ROCM-2. Sync: `GFX1201-PACKAGES-2026-09-13`.
Tajasarus: RX 9070 XT gfx1201, ROCm 10.0, assertions-enabled LLVM/MLIR
23.1.1, Ubuntu 26.04 WSL2. Source/compiler identities accompany this packet.

The driver now accepts explicit `package_native=True` for these adjacent
Graph→Schedule→Tile→Target→HSACO paths. Unsupported gfx1201 families refuse
instead of falling into the historical gfx1151 Graph packager.

- Static f16/f16→f32 matmul owns a conservative 16x16 register-staged profile.
  It does not inherit gfx1151's 32x64 LDS panel or its performance evidence.
- Scheduled f16/bf16→f32 forward attention carries exact architecture through
  the direct Tile adapter. It uses an explicit gfx1201 recompute policy,
  without inheriting gfx1151's saved-LSE threshold.
- Precomputed fragment addresses now include the architecture-owned half-wave
  K partition. Its absence silently loaded the wrong inputs on RDNA4 while
  remaining invisible on GFX11's replicated operand layout.
- Runtime registration and cached submission admit the same package ABIs.
  Tests remove cached launchers so each family proves independent startup.

`device-tests.txt`: **20 passed**. The corpus covers matmul aligned/ragged and
multi-K/multi-output-tile shapes; forward GQA, causal/noncausal attention,
optional f32 bias, and f16/bf16 storage; edited artifact rejection and driver
lineage. Forward causal references use the declared `max(Sk-Sq,0)` alignment.
Matmul tolerance is 2e-4; attention tolerance is 1e-2 relative / 1e-3 absolute
for the bounded low-precision softmax-product envelope. No precision-changing
optimization or performance promotion is authorized by these tolerances.

## Runtime attribution is still incomplete

`profiler.txt` records three passing independent matmul package executions
under `rocprofv3 --kernel-trace --hip-trace`. `profiler.db` is the original
ROCPD database; `profiler-counts.json` records its table counts. HIP activity
exists, but **kernel dispatches, kernel symbols and code objects each have
zero records**. Successful profiling-process exit is not kernel attribution.
Counter capability enumeration also remains blocked by WSL's missing `/dev/kfd`.
No kernel latency, overlap, calibrated clocks or selector promotion is claimed.

## Follow-through

Automatic public paired AD, resident saved-LSE/tape ownership, asynchronous
retirement, backward-program package admission, dynamic/epilogue and non-f16
matmul, broader attention envelopes, and sparse SWMMAC remain open. In
particular, this forward package is not a reusable training tape. Table 41's
sparse A dimensions are **after expansion**; a sparse producer must serialize
compressed storage and validated 2:4 indices separately from those dimensions.
The sibling gfx1151 profile is unchanged; this packet transfers no device
evidence to gfx1151, gfx1200, NVIDIA, Apple or x86.
