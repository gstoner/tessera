# Linked compiler-library identity

Owner FRONTEND-IR-MEDIUM-1 / E2E-REAL-6.
Synchronization key ROCM-LINKED-TOOL-IDENTITY-2026-10-07.

Compiler image cache identity now hashes the executable and loaded ELF
libraries. Version-query reuse watches dependency signatures, loader search
directories (including dependent-library RPATH/RUNPATH), loader environment,
and ld.so.cache. Discovery and content memoization are bounded to 32 entries.
An unchanged warm identity launches no subprocess. Non-ELF tool content
digest semantics remain unchanged.

95 host WSL checks pass, with 22 ROCm device cases skipped on the NVIDIA host.
Actual ELF fixtures prove library replacement, symlink retargeting and a new
higher-priority library invalidate identity without changing executable bytes.
The actual matching tessera-opt binds eight loaded libraries; its warm
identity check passes with subprocess calls forbidden. Shared diagnostic,
pass metadata and audit gates pass 307 checks.

Final gfx1201 numerical proof passes all 40 scalar primal/JVP cases.
All 16 alternating five-sample actual-package A/B profiles retain matching
compiler/toolchain fingerprints and reduce repeated subprocesses from 7 to 5.
Uncached/cached median ratios span 1.0972–1.1430,
with median 1.1265. These measure version metadata
reuse in native package wall time, not kernel speed or end-to-end warm JIT.

Raw samples, recorder/source/compiler fingerprints, host/device logs and
actual compiler linked-library hashes are retained. gfx1151 and broader
family/layout/AD envelopes require their own proof. General compiler closure
and reviewable PR delivery remain open.
