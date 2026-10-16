# SM120 typed saved-LSE reverse — 2026-10-16

Owner: E2E-REAL-6 / FRONTEND-IR-MEDIUM-1 / AD-RESIDUAL-EVAL-1.
Sync: SM120-SAVED-LSE-HALF-BACKWARD-20261016.

[Native compiler/kernel stage](native-stage/README.md) freezes 9b2159e655 with 639
focused and 167 owning cases. That direct-driver stage retains its own source scope.

[Frozen public owner proof](public-stage/README.md): Frozen 68c7e59300 passes 26,022
non-slow unit cases (5,902 skipped), 1,061 required production compiler-route cases and
339 owning RTX5070 cases. Both compiler-route and device proof lanes have zero
skips/failures; source/tool/provider witnesses remain unchanged. All twelve audit lanes
pass. Thirty-two matched stored-value characterization rows and thirty-two fresh
compiler-absent portable replays pass numerical checks.

Static half reverse uses typed Graph/AD/Schedule/Tile/NVIDIA/LLVM-PTX, checked native
ownership, FP32 correction and VJP v2. Half capture/JVP/dynamic/composed/storage/mask
and performance envelopes remain open. CI, review and delivery are tracked by the coordinating PR.
no aggregate five-slice closure is claimed.
