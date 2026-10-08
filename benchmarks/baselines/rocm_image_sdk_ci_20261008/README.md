# Pinned ROCm image SDK for portable compiler proof

Owner E2E-REAL-6 / FRONTEND-IR-MEDIUM-1.
Sync ROCM-IMAGE-SDK-CI-2026-10-08.

PR895's GitHub compiler-route selection built LLVM/MLIR 23.1.1 successfully,
then passed 110 tests and skipped 235 because AMD clang/device bitcode was
unavailable. The execution gate correctly refused that result.

The CI lane now downloads only the official ROCm 10.0.0rc4 core wheel, checks
SHA256 before installation/execution, installs into an isolated runner prefix,
and checks its TheRock and AMD LLVM source manifest. No GPU driver or GPU
device is required. The wheel's lib/llvm layout is aliased to the standard
llvm/ and amdgcn paths consumed by MLIR's native serializer and AMD clang.
The production Tessera compiler remains pinned to LLVM/MLIR 23.1.1.

The same official wheel has the owning Tajasaurus manifest: TheRock
16adc4d875fd4f65ea23c7c84e1c66706fde3047 and AMD LLVM
8f497e0992fb7513f7f78a6f6b6f1056c375e961. Source hashes, compiler hash,
archive hash and every device-bitcode hash are recorded. SDK tools reside
under scratch; the CUDA installation and system /opt/rocm are untouched.

## Validation

The actual installer and unchanged compiler-route selection run in Super-Bear
host WSL: **345 passed, zero skipped**. ci_require_executed accepts the
JUnit result with no skip exceptions. All 235 formerly skipped image cases
execute. This proves native image generation and contracts, not AMD device
execution on the NVIDIA host. Workflow/audit regressions pass 69 tests.

Two representative image tests initially failed at the native linker when
the wheel's inner LLVM directory was used as the toolkit root. The standard
toolkit aliases resolve that failure; the same two tests then pass. Both
logs are preserved. An invalid archive is rejected by its hash before any
installation, and no Python install directory is created.

The hosted GitHub runner must still confirm this change after publication.
Generic scaled_matmul batching/transpose closure remains open. Tests,
coverage states, no-skip gate and required-check aggregation are unchanged.

Official package location:
https://rocm.prereleases.amd.com/whl-multi-arch/rocm_sdk_core-10.0.0rc4-py3-none-linux_x86_64.whl
SHA256: 930c00c36aa67fd0b5fc5b59bc7078acecde07e6e27d97da56db2c9059d8551b
