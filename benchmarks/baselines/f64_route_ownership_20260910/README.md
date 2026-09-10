# F0 caller reconciliation and FP64 F2 migration

The census now includes the previously omitted x86 breadth module: 46 Graph
inputs, 14 scheduled inputs and 14 raw/unclassified entries. Caller candidates
resolve import aliases (including function-local imports), and local emitter
paths expose indirect construction within a backend module. Shadowed names,
reflective calls, cross-module helper chains and external callers still require
review. These are candidates and source identities, not execution certificates.

The reviewed x86 constructor remainder is:

- `package_matmul`: only uint8/int8-to-int32 VNNI; mixed signedness needs its own
  serialized operand contract. F32, BF16 and FP64 delegate to scheduled artifacts.
- `package_cohort2`: cohort-specific emitters and ABI contracts remain.
- `package_elementwise`: the retained generic Graph expression constructor.
- `package_graph_breadth` -> `package_abi` -> `emit_abi_tile_ir`: explicit breadth
  ABI composition remains; registry presence does not establish a Schedule owner.
- `package_attention_backward_semantics`: retained Graph semantic reconstruction.

NVIDIA/ROCm emitter paths are recorded separately in the census. No constructor
can be deleted merely because its input annotation or caller count changed.
Per-envelope certificates and actual runtime artifact ancestry remain required.

FP64 Graph -> Schedule -> Tile -> x86 now reaches the existing f64 native ABI.
Storage, accumulation and output are explicitly f64; descriptor projection and
Schedule replay reject altered fields/Tile text. The native compiler fixture
passes on Super-Bear's assertions compiler. Princess-Luna independently executes
(3,9,5) and (2,17,9) M/K/N shapes with float64 output and a 1e-13 oracle tolerance.
Six FP64/BF16 projection and numerical tests passed. No performance promotion or
ROCm GPU evidence is implied by CPU execution on that host.

371 registry, dtype, ABI and census tests passed in WSL. This increment advances
F0/F2; the other eight tracks retain their recorded gates. Full unit/lit suites
were not rerun for this increment.

The 11 audit tests, plan ownership check, Ruff, zero-error mypy ratchet and all
30 generated-document checks also passed.
