# Native bounded saved-LSE sequence images

Owner: E2E-REAL-6 / AD-RESIDUAL-EVAL-1.
Sync: NVIDIA-ATTENTION-BOUNDED-SEQUENCES-2026-10-08.

Graph MLIR now carries dynamic Sq/Sk with module capacity attribute
tessera.attention_shape_bounds = array<i64: B,Hq,Hkv,SqMax,SkMax,D,Dv>.
The native checkpoint verifier admits only positive fixed batch/head/width
axes and bounded sequence axes. Physical byte-address overflow and changed
fixed axes are rejected. Bounds participate in the sealed Schedule and LLVM
native contract; changing module bounds without updating the Schedule is
rejected. A fully logical bias follows runtime dimensions. Dynamic partial
physical bias requires a runtime extent carrier and remains unimplemented.

Initial proof: 60 native/static replay tests; one forward and one backward
image each execute six different sequence/input cases on an explicitly
queried RTX 5070 sm_120, including Sq=9/Sk=11 and asymmetric endpoints.
Independent float64 O/LSE/gradient comparison runs before and after every
timing window. Raw driver harness allocations use actual runtime dimensions.

Final matching build: 369 host WSL native/static replay, diagnostic/pass and
audit gates passed. Saved frontend Graph import is tested in both directions.
Twelve exact-device same-image cases passed after the final rebuild; maximum
absolute numerical error was 3.01833236e-07.
Initial fixture failures and build/test logs are retained alongside the final
source/binary-bound JSON packet.

## Boundary and remaining work

This packet proves native Graph/Schedule/Tile -> Target -> PTX execution.
The driver harness does not prove checked package ABI or public frontend/JIT
integration. Those guards, automatic AD export and generation-bound residual
ownership still require integration. No fallback arithmetic participates in
the device implementation. The oracle alone uses NumPy.

CUDA event windows include driver submission gaps. Completed host windows
measure resident launch plus synchronization and exclude compilation,
allocation, transfers and oracle work. Neither is an isolated kernel time,
and this packet makes no speedup or end-to-end public API claim.
