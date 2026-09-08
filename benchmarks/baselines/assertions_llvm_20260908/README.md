# Assertions-enabled compiler proof — 2026-09-08

Super-Bear WSL built upstream LLVM/MLIR 23.1.1 at
`e7ce3600b55034ddf819638f395e3c475fad5be2` in an isolated toolchain directory.
See `scripts/build_assertions_llvm.sh` and the COMPILER-DEVEX-1 recipe in
`docs/audit/compiler/COMPILER_REFACTOR_PLAN.md`.

`probe.json` records an executed LLVM SmallVector assertion (SIGABRT).
`tessera.json` records the isolated consumer compiler hash and 257 passing
focused tests, with five target-tool skips. The upstream no-RTTI build requires
matching `-fno-rtti` in this Tessera configuration; compilation ends in
`-UNDEBUG` even in RelWithDebInfo.

The initial lane aborted because NativeTapeToGPUPass loaded Tile during pass
execution without declaring its dialect dependency. Adding TesseraTileDialect
to dependent dialects fixes the defect. Production LLVM was not replaced. This
is focused compiler-contract validation, not full-pass-corpus, installed-driver
or physical-backend closure.

A further 76 composed tape/ANN/storage/reader tests pass with assertions-ON
Tessera. Their downstream packaged LLVM tools and host JIT remain release
builds. For ROCDL image-only cross-compilation on the NVIDIA host, set ROCM_PATH
to an isolated directory containing `llvm/bin/ld.lld` linked to the existing
`/usr/lib/llvm-23/bin/ld.lld`; this supplies no ROCm runtime or device proof.
