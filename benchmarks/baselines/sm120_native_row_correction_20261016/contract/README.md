# Native attention row-correction tensor contract

Owner: E2E-REAL-6 / AD-RESIDUAL-EVAL-1 / FRONTEND-IR-MEDIUM-1.
Synchronization: SM120-NATIVE-ROW-CORRECTION-20261016.

First implementation phase only: the internal tensor operation is registered
with pure effects and verifies static half Q/K/V, independently half/F32 dO,
physical F32 bias broadcasting, F32 LSE/seed/result, exact operand roles and
closed numerical policy. The production compiler passes 31 new contract cases
and 30 adjacent registry/dtype cases (61 total, zero skips/failures). The parent
compiler refuses this new operation as unregistered.

record.json binds the actual source files, compiler and logs. Frozen sources
retain this phase when later implementations change the checkout. This is
parser/verifier evidence, not a numerical kernel, executable package, public
operation promotion, Schedule/Tile closure, device proof or performance result.
Native product projection, workspace producer/consumer lowering, runtime ABI,
portable program integration and exact-device benchmarks remain in progress.
