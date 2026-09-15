# macOS 27 Apple validation — 2026-09-14

Owning host: Apple M1 Max (Apple7), macOS 27.0 build 26A428. All GPU probes ran
unsandboxed under the user's standing authorization for Mac Metal validation.

- `language_probe.json`: runtime compilation and execution with MSL 4.0 and 4.1,
  for float, half and bfloat. Each checks 1.5 + 2, 1.5 * 2 and 1.5 / 2 exactly.
  This is a basic arithmetic smoke test, not exhaustive rounding, underflow,
  matrix-accumulation or hardware-instruction proof.
- `lowp_tests.txt`: 79 passed in the existing native low-precision suite, using
  a fresh runtime built explicitly against Xcode's macOS 26.5 SDK. Includes
  fp16/bf16 softmax and attention backward, shape/bias and contract checks.
- `offline_metal41.txt`: active Xcode 26.6 Metal compiler rejects `-std=metal4.1`.
- `sdk27_build.txt`: rebuilding the current runtime using the installed CLT27
  compiler and SDK27 fails on the previously excluded microscaling block.
- `environment.json`: OS/toolchain identity and runtime source/library hashes.

Reproduce the language probe with the CLT27 clang++, an explicit SDK27 sysroot,
Foundation and Metal frameworks, then run the executable outside the sandbox:
`benchmarks/apple_gpu/probe_metal_language.mm`.

The regular runtime build uses `-std=c++17 -shared -fPIC -O2 -fobjc-arc`,
`-isysroot /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX26.5.sdk`,
and the Foundation, Metal, MetalPerformanceShaders and MetalPerformanceShadersGraph frameworks.
Run `tests/unit/test_apple_lowp_native_contract.py` with `PYTHONPATH=python:.`,
`TESSERA_OPT` pointing to the native compiler and `TESSERA_APPLE_GPU_RUNTIME_LIB`
pointing to that fresh dylib.

No FP8/FP4/MX GPU execution, acceleration, fleet re-seal or performance promotion
is claimed. Existing sealed packets remain historical evidence for their own
OS/runtime identity. The production runtime source was not changed here.

The SDK26.5-built runtime returns 0 from
`tessera_apple_gpu_supports_microscaling`, even on this upgraded OS, as expected
from its compile-time gate. After host connectivity was restored, Tajasarus WSL ran the documentation
audit tests: 11 passed; documentation lint passed. No Mac substitute was used
for those non-Metal project tests.

## SDK27 API repair

The runtime now builds with CLT27 clang and an explicit MacOSX27.0.sysroot
(`-isysroot /Library/Developer/CommandLineTools/SDKs/MacOSX27.0.sdk`).
All seven original errors are fixed. SDK27 low-precision suite: 79 passed;
opt-in descriptor suite: 3 passed; reference/Metal bridge on Tajasarus WSL:
37 passed. Descriptor tests use `TESSERA_SDK27_DESCRIPTOR_LIB` pointing to
a freshly built library. Descriptors do not establish native FP8/FP4 execution.

Downloaded Apple Metal component 27A5194o with
`xcodebuild -downloadComponent MetalToolchain -buildVersion 27A5194o`.
Its compiler 32023.917 successfully compiled the simple float probe with
`-std=metal4.1 -c`. A local copy is at
`~/.local/share/tessera/toolchains/Metal27.xctoolchain/usr/bin/metal`; invoke
that binary explicitly until the Xcode update supplies the default compiler.
Default `xcrun metal` was still 32023.883 during validation. No global
developer-directory setting was changed.

The original environment and error logs above are pre-fix evidence. The repaired
runtime has not yet been committed/re-sealed for the revision-bound fleet packet.
No new fleet or performance promotion claim is made.

## Xcode27 default-toolchain validation

After upgrading to Xcode27.0 build27A266a, installed its matching Metal component
with `xcodebuild -downloadComponent MetalToolchain`. Default `xcrun metal` now
reports 32023.921 (metalfe-32023.921.6), and `-std=metal4.1` compilation succeeds
without an override. The earlier extracted compiler is no longer needed.

Rebuilt the runtime using `xcrun clang++` and Xcode's SDK27 explicit sysroot;
zero errors, one existing MTLGPUFamilyMac2 deprecation warning. Fresh runtime
validation: **82 passed in 20.07s** across `test_apple_lowp_native_contract.py`
and `test_apple_sdk27_descriptor.py`. This supersedes the toolchain blocker
above, not the remaining FP8/FP4 execution or fleet re-seal obligations.

## Packed-buffer numerical execution

`tests/unit/test_apple_packed_numeric.py`: four tests pass with a fresh runtime
on this M1 Max. Set `PYTHONPATH=python:.` and
`TESSERA_PACKED_NUMERIC_LIB=/path/to/fresh/runtime.dylib` to reproduce.
The explicit `tessera.compiler.apple_packed_numeric.evaluate` entry binds packed
E4M3/E5M2/E2M1 buffers to Metal native pack/unpack and fp32 arithmetic. Coverage
includes every code, signed zero, subnormal values, exceptional values,
nearest-even midpoint conversion and format-specific overflow. Negative FP8
NaNs canonicalize during unpack; no NaN sign/payload guarantee is made.

The new ABI returns success only after GPU completion and copies results back
only then. A failed or timed-out submission never runs a CPU reference. This
bounded numerical lane is not generic MX, MTLTensor/matmul binding, selector
admission or a performance claim. The generic microscaling flag remains false.

## Matrix operands, accumulation, matched kernels and fused epilogues

Same host, Xcode 27.0 (27A266a), Metal compiler 32023.921, runtime rebuilt via
`ninja -C build TesseraAppleRuntimeShared` against the SDK27 sysroot. All GPU
runs unsandboxed under the standing Mac Metal authorization.

- `tests/unit/test_apple_gpu_lowp_matmul2d.py` (68) + `test_apple_gpu_lowp_accumulation.py`
  (31): **99 passed**; the existing `test_apple_gpu_metal4.py` (76) still passes
  with the gelu fix. Neighbouring suites: 110 passed (lowp native contract,
  packed numeric, SDK27 descriptor, Metal bridge, governance).
- `lowp_matmul2d.json`: matched device timings (Metal 4 counter heap,
  `timing_source = metal4_timestamp_heap`, median of 12 after 3 warmups) for
  MPP fp16, MPP e4m3 / e5m2 / e2m1 / half x e4m3, the `simdgroup_matrix` f32
  kernel, MPS fp16 (wall only), fused vs decomposed bias+gelu, plus host
  packing cost per shape. Rows carry `route` (Decision #12 amendment).

Findings: FP8/FP4 `matmul2d` is emulated on the M1 Max (0.77-0.93x fp16 at
1024^3-2048^3); accumulation is at least fp32 and behaves sequentially at the
2^30 + 256 probe; Apple enforces 128-byte row strides and 128-byte data-plane
offsets for 8/4-bit tensors and applies 4-bit buffer offsets at 2x (worked
around, canaried); fast-math gelu produced NaN from pre-activation 10.25 in
every fused epilogue (fixed). Fused epilogues are 0.5-5% slower on device time
than decomposed; no routing change. Not block scaling, not fleet re-sealed,
not a performance promotion.
