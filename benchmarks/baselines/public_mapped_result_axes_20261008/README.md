# Public compiler-owned mapped result placement

Owner FRONTEND-IR-MEDIUM-1 / AD-RESIDUAL-EVAL-1 / LAYOUT-ALG-1.
Sync MAPPED-RESULT-EXECUTION-20261008; depends on PR904.

Public vmap over a typed gfx1201 JIT projects out_axes as semantic Graph
transpose. Native MLIR verification/AD, Schedule, Tile, GPU Target and native
image packaging carry that movement into the checked HIP program. Nested
maps compose output axes in their level order. No Python output transpose is
used for execution; NumPy transposes appear only in independent certificates
and expected-value tests.

RX9070XT/gfx1201: 20 public device tests pass, including FP8 floating-scale
and MXFP8 E8M0 primal, shared/independent operands, negative axes, nested
primal/scale-JVP and warm changed-input/seed reuse without compiler calls.
605 map/registry tests and four existing RTX5070 attention JVP tests pass.
These RTX cases are sibling regression evidence, not mapped-output migration.

gfx1201.json records six correctness-gated public profiles and live inventory,
matching compiler/runtime/source digests, complete member operation lists,
captured per-member samples and cold/warm public call timing. Scale JVP has
three products, one tangent sum and two result movements. Event values are
already averaged over repetitions. Captured graph windows include device
dispatch but exclude capture, instantiation and copies; they are diagnostic
grouped-member timing, not ordinary interleaved program timing. Warm public
calls include frontend/cache admission, host input preparation, native
execution and copied returns. No speedup/default-route promotion is claimed.

Reproduce: benchmarks/rocm/record_public_mapped_result.py --output
<scratch>/packet.json on the owning gfx1201 host, using matching native tools
and the complete native movement/program runtime provider.

The diagnostic provider from PR904 is sufficient for direct PreparedScaledProgram
profiling. Public runtime loading also requires the movement/NVFP4 ABI symbols;
the owning proof builds native_program_runtime.cpp, native_movement_runtime.cpp,
native_nvfp4_runtime.cpp and native_image_cache.cpp into the complete provider.

Nonleading reverse AD requires inverse-cotangent ownership/export integration.
Dynamic/strided results, encoded-scale/storage derivatives and broader
NVFP4 maps remain open. Generic primitive closure is not claimed.
