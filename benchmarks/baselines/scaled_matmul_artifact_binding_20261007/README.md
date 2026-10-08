# Native repeated-product artifact binding — 2026-10-07

Owner: FRONTEND-IR-MEDIUM-1 / E2E-REAL-6.
Synchronization key: SCALED-MATMUL-ARTIFACT-BINDING-2026-10-07.

Schedule content hashes remain unchanged. Each native matmul instance receives
a module-local binding on its Graph subject, Schedule carrier and artifact.
Tile consumes exactly the matching hash/binding instance and retires that
artifact. An absent binding preserves the checked legacy single-product route;
a mixed, empty, non-string or altered Graph/Schedule binding is rejected.
Forged Graph and artifact bindings have negative native fixture coverage.
The JVP builder drops both derived hash and binding before rescheduling.

The scale-seed JVP now reaches Tile and ROCm Target IR with unchanged scale
group K128/fp32 and exact-per-block partial accumulation. The full native core
lane passes 489 fixtures, with 66 unsupported features (555 discovered).
The first focused Tile check failed only because its FileCheck attribute
order was wrong; the final full lane includes the corrected positive fixture.
Initial failing receipt is retained.

This is artifact evidence, not executable multi-product AD proof.
ROCm Target records still have the same parent name for each term and the
program retains tessera.add. Native multi-kernel symbol/argument projection,
sum lowering, allocation lifetime and executable program ownership remain
open. The general transposed FP8 Schedule route remains outside admission.
The earlier generic batching/linear-transpose full-unit failures are unchanged.
No new AD numerical or timing result is claimed. Owning GPU regressions of
previously executable NVFP4/attention routes are recorded separately when done.

Changed source and rebuilt compiler binaries are content-bound in SHA files.
Historical JVP packet remains valid for its earlier recorded source/tools.

Both native backend lanes pass all 153 fixtures. Matching rebuilt tools pass
159 RTX 5070 NVFP4/attention numerical tests (140.88 seconds, no skips).
This is regression evidence for existing executable routes, not scaled AD.
The native multi-product binary experiment refuses a duplicate GPU module
symbol. The generator also owns whole-wrapper retirement; merely renaming
kernels would not preserve sum/program semantics. Implement checked native
program ownership before promoting this AD route. The failure is retained.
