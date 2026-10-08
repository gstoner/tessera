# Compiler orchestration and tool-identity performance

Owner FRONTEND-IR-MEDIUM-1; sibling E2E-REAL-6.
Synchronization key: NATIVE-COMPILE-ORCHESTRATION-2026-10-05.

## Changes and authority

Saved-LSE attention JVP runs Graph-to-Schedule and Schedule-to-Tile in one native MLIR pass manager. Typed SSA and normal intermediate verification are retained; a Python subprocess and printed/reparsed Schedule boundary are removed.

Native GPU packaging uses the shared exact SHA-256 file-identity helper. It streams bytes through native hashlib on first identification and reuses that digest only for the same resolved path/device/inode/size/mtime/ctime. At most 128 identities remain per process; access is locked. Both the opened file and pathname must retain the same identity during reading. Atomic replacement, same-size edits with restored mtime, and symlink retargeting invalidate reuse. A concurrent rebuild is rejected. Digest fields and checked ABI/image provenance remain exact SHA-256.

Python remains the frontend and thin wrapper. There is no package/image cache in this change: every benchmark arm constructs a fresh package, compiles an image, validates its output-writing body, lowers/links its native scratch-size companion and binds its identity. The bodyless-image guard is unchanged; its old stale explanatory comment is corrected.

## Measurements

Super-Bear RTX 5070 / sm_120; matching LLVM 23.1.1 assertions compiler and CUDA 13.3 host. Scope is compiler wall time, not GPU throughput.

Initial balanced ten-trial Graph/Schedule/Tile comparison: split processes median 17.85 ms; one native pass manager 11.13 ms. Tile IR is byte-identical. Complete-package A/B initially shows no consistent improvement from this alone (roughly 0.9 seconds).

cProfile attributes about 0.44 seconds to CUDA image disassembly validation, and 0.19 seconds to tool reads plus hashes. The latter is eliminated on stable-identity reuse; the former remains required.

Five balanced trials for each Sk=5/129 arm compare:
- split native passes plus uncached exact tool hashes;
- one native pass manager with cleared identity cache;
- one native pass manager with stable identity reuse.

All thirty fresh packages have identical image, arena IR, entry, sizer and ABI for their shape. Cold identity means a cleared in-process identity memo; it does not mean a cleared OS filesystem cache. Samples, tool/implementation/recorder hashes and stage timings are retained. Final measured reductions are 11.6–14.3% for cold identity and 23.0–23.2% with reuse.

| Sk | Split / uncached identity | Native / cold identity | Native / reused identity |
| --- | ---: | ---: | ---: |
| 5 | 946.79 ms | 811.84 ms | 728.97 ms |
| 129 | 890.69 ms | 787.31 ms | 683.96 ms |


## Validation and remaining work

321 focused tests pass, including in-place/atomic/alias rebuild and read-time mutation cases. Shared identity/runtime/arena/stream/diagnostic tests: 142 pass, one unrelated skip. Ten ordinary JIT attention cases pass independent finite-difference/forward oracles on RTX 5070. Thirty tool-identity and native NVFP4 ingest/package/JIT tests pass on RX 9070 XT / gfx1201. No gfx1151 or Metal execution/performance claim.

A native persistent compiler session and native package/image inspection are the next measured architectural targets. General frontend/AD and the broader five-slice objective remain open. FP8/MXFP8/MXFP4 numerical/quality/performance gates are unchanged. Graphify CLI is absent from the authoritative scratch checkout; no refreshed graph is claimed.
