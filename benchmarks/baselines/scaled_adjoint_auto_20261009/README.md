# Native FP8 scale-adjoint automatic recipe — gfx1201

Owner E2E-REAL-6 / AD-RESIDUAL-EVAL-1 / FRONTEND-IR-MEDIUM-1.
Synchronization ROCM-SCALE-ADJOINT-AUTO-20261009; dependent on PR910.

## Engineering change

The RHS scale-gradient reduction is the dominant member of the named
7x19x256 two-output program: about 1.63 ms versus 0.31 ms for the LHS scale
gradient. Requested return ordering does not change this attribution.
The existing native wave recipe partitions column contributions and preserves
each original K-group dot, with compensated local FP32 accumulation and a
32-lane XOR join. Automatic choice now lives in Graph-to-Schedule, derived from
the actual scalar reduction SSA. It chooses a wave for byte-coefficient scale
gradients with <=128 outputs and contribution span >=4. It reads the RHS scale
block span, rather than guessing from logical N. Continuous f32 adjoints,
narrow groups and larger frames retain serial geometry. The selected algorithm,
workgroup width, body and checked geometry remain hash/replay bound.
Python only passes explicit policy intent and retains schedule-keyed caching.

TESSERA_ROCM_SCALE_VJP_SCHEDULE=auto selects this policy. The default remains
serial_per_scale_element. This explicit recipe does not change any primal
numeric policy, encoded-scale differentiation boundary or sibling schedule.

## Proof

Fresh full matching LLVM/MLIR23.1.1 assertion compiler build completed for
tessera-opt and tessera-rocm-opt, including enabled x86, NVIDIA, EBM and Clifford
libraries. The first native API build failure is retained in scratch; the
region-access repair rebuilt successfully. The compiler, shared layout library,
source and proof delivery archive SHA256 is
8a49df74ae6f847b98d2fac4722b32c6ae99ed2371c44a4d8f6f432786005801;
the owning host verified it before extraction and reported LLVM23.1.1 assertions.

Host WSL: 25 native auto-selector tests, 401 existing native export/lowering/map
regressions, and 298 diagnostic/pass metadata/cache admission checks pass.
The selector tests cover narrow RHS blocks despite wide N, large LHS versus
small RHS output frames, byte versus continuous operands, conflicting options,
sealed Schedule corruption, actual ELF packaging and checked geometry.

RX9070XT/gfx1201: 76 public auto-policy tests cover sharing, KN/NK RHS,
selected/reordered roles, narrow-column serial selection, independent numerical
comparison, changed-seed warm compiler exclusion and retained outputs. Four of
these verify continuous four-gradient serial parity. Another 48 checks cover
FP8/MXFP8 mapped primal/JVP and NVFP4 ingest/MXFP4 producer-consumer regression.
Owning pytest retains its existing unknown-timeout-config warning.

Delivery gates: CI-scope Ruff and zero-error mypy pass; all 32 generated
documents are in sync, and 25 audit/recorder gates pass. Full CI unit selection:
19,920 passed, 9,583 skipped, and two existing generic scaled_matmul
batching/transpose closure failures in 350.46 seconds. Native compiler-specific
tests are separately proved above; their host-free CI skips are not device proof.
Graphify AST update completed after the source edits.

## Measurements

Reproduce benchmarks/rocm/record_scaled_adjoint_members.py --paired --auto
--output PATH with matching compilers/provider. Twelve cases use seven rounds
rotating/reversing arm order, with independent numerical checks before and after
every ordinary/captured window. All arms share identical native program JSON;
auto and explicit wave images match exactly in these named cases.
The packet records live inventory and compiler/runtime/source/oracle/recorder
identities. Twelve local source/compiler identities match the owning packet.
Captured member windows include device graph dispatch; ordinary native program
events include host dispatch gaps. Neither includes compilation, uploads,
readback or complete public-call latency. The table is ordinary event medians.

| M,N,K | Requested scales | Serial ms | Wave ms | Auto ms |
| --- | --- | ---: | ---: | ---: |
| 3,5,37 | sa | 0.020646 | 0.005589 | 0.005646 |
| 3,5,37 | sb | 0.137357 | 0.013092 | 0.013518 |
| 3,5,37 | sa/sb | 0.159391 | 0.019554 | 0.019798 |
| 3,5,37 | sb/sa | 0.156683 | 0.019587 | 0.019269 |
| 7,19,256 | sa | 0.308063 | 0.010852 | 0.011309 |
| 7,19,256 | sb | 1.629561 | 0.050863 | 0.050818 |
| 7,19,256 | sa/sb | 1.937842 | 0.062629 | 0.062010 |
| 7,19,256 | sb/sa | 1.938824 | 0.061924 | 0.062659 |
| 17,19,256 | sa | 0.286711 | 0.010830 | 0.011217 |
| 17,19,256 | sb | 3.984080 | 0.115613 | 0.115493 |
| 17,19,256 | sa/sb | 4.253080 | 0.126692 | 0.125908 |
| 17,19,256 | sb/sa | 4.263201 | 0.125880 | 0.126742 |

The 7x19x256 two-output auto program is about 31x faster in this native event
domain; this is not a complete public-call speedup. FP8/MXFP8/MXFP4 primal
regressions establish retained correctness, not a format-wide performance claim.

## Remaining work

Widen sharing/orientation/cardinality/performance controls before default
promotion. Continuous adjoint wave admission, generic batching/transpose
closure, broader composed/dynamic AD and sibling exact-device proof remain
open. This slice does not complete the original five-program objective.
