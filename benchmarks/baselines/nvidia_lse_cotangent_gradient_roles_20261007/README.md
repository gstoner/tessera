# Seeded compact gradients and physical bias reduction

Owner NVIDIA-LSE-1 / AD-RESIDUAL-EVAL-1 / E2E-REAL-6.
Synchronization key NVIDIA-LSE-COTANGENT-GRADIENT-ROLES-2026-10-07.
Publication pending.

Native checkpoint Graph/Schedule/Tile now packages the explicit f32 row-LSE
cotangent with compact activity or complete physical bias gradients. Sealed
compact symbols encode the active roles, bias, launch layout, thread count,
Schedule digest and seed marker. The C++ bridge validates that marker,
includes the row pointer before outputs and sizes allocation/argument arrays
for all roles. Complete seeded bias-gradient symbols preserve their explicit
gradient marker. No Python production Graph/Schedule/Tile constructor or
production kernel loop is added; diagnostic authored MLIR fixtures exercise
the existing native checkpoint carrier with four logical result roles.

The checked seed ABI shares the package/common-runtime replay path. Its
physical contract delegates compact role projection to the existing compact
validator. Numerical scale, causal/end-aligned policy and physical bias
reduction bind the saved checkpoint identity. Invalid activity, role, seed,
thread, guard, entry and numerical-policy mutations are refused. Ordinary
unseeded compact parsing and ABI remain compatible.

Exact RTX 5070 proof: all 15 nonempty compact masks, packed/logical launches,
64/128-thread blocks, grouped heads and causal/full cases pass. Complete
bias gradients cover full, column-broadcast and row-broadcast physical
storage. Pure row seeds exercise zero dV and nonzero bias derivatives.
Host, resident, event and resident-event launches compare every physical
output with independent FP64 derivatives, including explicit broadcast-axis
reduction. JSON replay retains only active physical outputs. The final
combined seeded gate has 119 passed (75 role tests plus 44 previous package
cases); checkpoint/registry regressions have 496 passed; legacy compact has
39 passed. Counts overlap earlier receipts. Matching core/NVIDIA compiler
and runtime build passes; both physical-contract Mypy checks and Ruff pass.

Recorder: benchmarks/nvidia/record_lse_cotangent_gradient_roles.py.
The recorder reuses benchmarks/nvidia/record_lse_cotangent_package.py for
correctness-gated timing. rtx5070.json has 66 rows, 330 CUDA-event and 330 host
end-to-end windows, all poisoned then numerically checked. Event timing uses
C++ launch loops with transfers/readback outside events. Host timing includes
checked descriptor validation and bridge copies. Source and compiler/runtime
hashes are verified. No background graph/docs/test job was active in this
characterization run. Timings characterize this static envelope; no default
promotion or broader workload speedup is claimed.

Remaining: automatic multi-result attention AD must generate and carry the
row seed; private residual ownership and public JIT AD integration remain
open. General dynamic/composed attention and full batching/transpose closure
also remain open. This proof is for the typed native checkpoint consumer,
not automatic differentiation of a public two-result frontend function.
Apple/x86/ROCm obtain no new physical admission or device parity.
