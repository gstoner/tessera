# Checked bounded saved-LSE packages

Owner E2E-REAL-6 / AD-RESIDUAL-EVAL-1.
Sync NVIDIA-ATTENTION-BOUNDED-PACKAGES-2026-10-08.

Native checkpoint Graph products with dynamic Sq/Sk and fixed B/head/width
dimensions lower through Schedule/Tile to one forward/backward image pair.
The capacity [1,2,1,9,11,8,6] is part of the sealed identity; package decoding
preserves the native MLIR 23 dynamic sentinel. Min/max guards bind every
dynamic tensor axis, and role checks bind buffers to actual launch scalars
before loading CUDA.

A serialized pair captures private Q/K/V, O/LSE and gradient allocations for
one actual runtime shape. Tests retain multiple differently shaped frames
and old gradients across new seeds. Synchronous and asynchronous execution,
full bias, compact dQ/dV and LSE cotangents are checked independently. Invalid
capacity, cross-buffer roles and removed guards fail before driver loading.

173 initial host package/static regressions, 72 compact/seeded/pre-driver tests
and 20 RTX 5070 owning tests pass. The final aggregate lane passed 581 host
WSL tests and 20 owning-device tests after ABI isolation and fixture repairs.
Initial decoder/guard/fixture failure logs are retained.

Eight checked-API timing arms compare two runtime shapes, two bias policies
and sync/async modes with independent float64 numerical checks. Images remain
identical across shape and ownership policy for each bias setting. Capture
host time includes module loading, allocation, snapshot and forward completion.
Backward event windows include allocation/copy/kernel/free and submission
gaps; completed host includes synchronization. No isolated kernel or
counterbalanced speedup/default promotion is claimed.

## Remaining integration

This is explicit verified scheduled Graph product packaging and resident
execution. Public JIT symbolic tracing, automatic native paired AD export,
dynamic partial physical-bias carriers and arbitrary attention composition
remain open. The raw native-image packet is preserved separately; its evidence
alone is not used to claim package ABI completion.

Final eight-arm benchmark maximum absolute error: 6.41649861e-08.
