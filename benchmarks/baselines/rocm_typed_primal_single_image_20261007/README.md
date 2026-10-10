# Typed primal native image binding — gfx1201

Owner FRONTEND-IR-MEDIUM-1 / ROCM-FP8-BLOCKSCALE-1.
Sync TYPED-PRIMAL-SINGLE-IMAGE-2026-10-07.

Scalar and coupled-batch typed FP8/MXFP8 packages bind the actual native
Graph-derived member image, entry symbol, scalars and launch geometry.
The prior adapter separately compiled a Tile image for the descriptor while
the native program executed a second Graph-derived image. The duplicate
compilation has been removed. Native Graph/Schedule/Tile/Target/LLVM remains
the production path; Python binds semantic types and the checked ABI.

Host WSL gates pass 388 cases, including 16 scalar/coupled combinations:
unbatched, shared-RHS rows, independent RHS and shared LHS, each with both
B orientations and FP32/E8M0 scales. Image payload, entry, geometry, scalar
extents and original operand roles match the native program.
Exact gfx1201 device gates pass 309 cases, including scalar/coupled maps and
independent prefixes, numerical oracles and changed-input compiler-free reuse.
The device receipt retains a pytest timeout-plugin configuration warning.

## Package wall-time attribution

The recorder uses the actual compiler and alternating three-sample A/B.
Graph-to-Tile derivation precedes the timed region. Empty-cache samples clear
both legacy Target and native-image dictionaries before each arm.
Sixteen empty-cache profiles have control/candidate ratios 1.707–1.788,
median 1.759; subprocess count drops from 11 to 7.

A subsequent reused-cache phase has ratios 0.879–0.919, median 0.901.
Its first control sample may still be cold because the preceding candidate
cleared the shared dictionaries; raw subprocess counts retain both 5 and 11.
Later control samples use five subprocesses; candidate samples use seven.
The cache-phase medians therefore expose remaining metadata/fingerprint work,
not a universal package or execution speedup. The initial unqualified cold
recorder result is superseded by these explicit phase measurements.

Compiler SHA256:
25a4f542b0997ae4192d4e018830fff3aa78f7634a72e20ee1010cd2b9602b31.
Actual adapter hashes and sample arrays are in package-cost.json.
The preserved control is the prior adapter, not a reimplemented approximation.

## Open scope

Reused-package metadata overhead, scalar transposed-A/partial-group admission,
general dynamic/nonleading/composed/storage AD, generic batching/transpose
closure, fresh full-suite green and PR delivery remain open.
ROCm proof does not establish Apple/x86/NVIDIA physical parity.
