# Compiler-projected native scaled-product AD package

Owner: FRONTEND-IR-MEDIUM-1 / E2E-REAL-6.
Sync: SCALED-MATMUL-NATIVE-PACKAGE-2026-10-07.
Exact-device architecture: gfx1201, AMD Radeon RX 9070 XT, Tajasaurus.
Compilation host: Super-Bear WSL, LLVM/MLIR 23.1.1, ROCm cross-target SDK.

## Compiler and runtime contract

Native forward AD exports a machine-readable SSA program tied to the actual
differentiated Graph witness, typed buffer storage and exact lifetimes.
Native selected-member projection carries actual products/sum through
Graph -> Schedule -> Tile -> ROCm Target -> ROCDL/LLVM -> HSACO.
The backend generators emit actual kernel entry symbols, scalar ABI and
resolved grid/block geometry. Package orchestration reconstructs no Graph,
Tile arithmetic or numerical kernels. Package validation checks member/program
lineage, prefix SSA bindings, output ownership, capacities and scalar extents.
One native C++ owner retains image leases and device storage and executes
the complete sequence; Python performs no per-member production launch loop.

package_native_scaled_jvp accepts a textual native Graph AD request.
PreparedScaledProgram marshals its compiler ABI into the prepared HIP owner.
This API is not yet connected to ordinary public JIT AD capture.

## Validation and measurements

The native package/registry gates pass 305 tests, including adversarial
architecture, scalar, future-input, geometry-overflow, witness, output and
non-ELF mutations. The owning GPU replay disables subprocess.run to prove
compiler-free warm execution. All four images execute from compiler-projected
metadata, including the hash-derived sum symbol. Changed tangent inputs
invalidate prior outputs and pass the independent oracle after rebinding.

Static ragged scale-only case: M17/N19/K256.
Primal/tangent maximum absolute errors:
1.9878801e-06, 1.7086885e-06.
Central scale finite-difference maximum error:
1.6137993e-06.
Eleven device windows of 100 complete native sequences:
median 0.02927467 ms per sequence.
Warm invoke plus two readbacks: median
0.38476498 ms.
Update/invoke plus two readbacks: median
0.48252195 ms.
The HIP event window includes native C++ enqueue gaps across four kernels;
host windows exclude compilation/preparation. No isolated-kernel speedup
or selector promotion is claimed.

## Evidence and remaining work

projected-package.json records native Graph/SSA and physical member manifests,
images and target. projected-package-device.json records package/image/runtime
hashes and all timing samples. native-owner-inputs.sha256,
runtime-cmake-toolchain.txt and rocminfo.txt identify the owning runtime build
and GPU. validate_projected_package.py is the diagnostic replay/oracle driver.

Ordinary public JIT AD integration, transposed FP8 Schedule admission,
general scaled-matmul batching/linear transpose, sibling physical program
integration and the red full-unit gate remain open.

Final native core/NVIDIA/ROCm regression lanes pass 644 fixtures with 66
unsupported feature cases. Current compiler reproduces the exact device-tested
program/member manifests and all four images; compiler-tools.sha256 binds
the final tool binaries. Audit/citation/routing gates pass 22 tests.
