# Writable source views and exception-value transport

Owners: W4-PRODUCT-1 / AD-RESIDUAL-EVAL-1 / W2.4a.
Sync key: SOURCE-VIEW-EXCEPTION-2026-09-09.

Native CPU tests cover ordered writes through overlapping contiguous rank-one
views of an explicit containing input, rejection of changed view offsets,
single-element f32 exception payloads crossing loop/finally completion, nested
re-raise, mutable-exception-alias refusal, and object-field state VJP without
copyback. The focused lane passes 102 tests with the assertions-enabled LLVM 23
compiler and native CPU JIT. Package mypy passes. validation.json binds the
source/test hashes; this is a bounded execution contract, not general Python
object, view or exception closure.

nvidia_regression.json and rocm_regression.json independently recheck the
existing single-state asynchronous update consumer on SM120 and gfx1151.
They do not prove multi-input writable views or exception payload execution on
GPUs. The source producer's containing-input requirement does not discover
hidden NumPy storage or silently materialize an unrelated backing array.
No performance promotion.

Final checks: 248 audit/governance/diagnostic/pass-metadata tests and all 30
generated-document checks passed; Ruff and package mypy passed. The full unit
suite passed 18,568 tests (2,266 skipped, 870 deselected) before the final
alignment rejection. The 102-test focused rerun covers that guard; the alignment regression also
passed independently on Princess-Luna. Final GPU packets bind the updated source.
