# Native automatic sparse selection — 2026-09-14

Owning device: Tajasarus gfx1201, ROCm 10.0, WSL2, LLVM 23.1.1 with assertions.

`JitFn.compile_sparse_auto` emits one native Graph/Schedule/Tile artifact for
isolated fp16/bf16 matrix multiplication. Each K=32 tile uses a wave-wide 2:4
agreement before branching: SWMMAC when eligible, native dense accumulation
otherwise. No pruning or CPU fallback occurs. Selection is an explicit policy;
default dispatch and calibrated promotion remain separate.

Device tests reuse one image for dense, sparse, mixed (one invalid lane), and
changed-density inputs. The fp16 AD-configured parent also executes native
backward against the logical derivative oracle. This is not arbitrary AD
composition or a performance claim.

462 focused tests passed; Ruff and focused mypy passed. The test output is
`focused_tests.txt`. Broader regions, default promotion, and arbitrary
control-flow/higher-order AD remain open.
