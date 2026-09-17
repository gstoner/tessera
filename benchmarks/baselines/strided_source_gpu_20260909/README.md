# Strided source GPU execution

Owners: W4-PRODUCT-1 / AD-RESIDUAL-EVAL-1 / W2.4a.
Sync key: STRIDED-SOURCE-GPU-2026-09-09.

Independent SM120 and gfx1151 packets execute local positive-stride rank-one
aliases through one owned root, with asynchronous checked copyback and active/
expired-reader refusals. Copies precede writes during bufferization. Known
subview/cast chains resolve to private allocations or declared outputs;
input-rooted writes still refuse. Additional copies can increase temporary
storage; this is not performance promotion.

332 focused source, native ownership, status, diagnostic and pass metadata tests
passed on Super-Bear using its rebuilt assertions-enabled LLVM 23 compiler.
Both hosts rebuilt tessera-opt. Ruff and package mypy passed. CPU tests cover
positive-stride writes, local slice capture, exact-alias root gradients and
explicit static causes/from-None suppression. Exact aliases accumulate at the
canonical root; duplicate alias gradients are zero.

Remaining: negative/multidimensional view maps, general slice adjoints, caught
exception identity/implicit context/tracebacks, dynamic causes and GPU exception
transport. These packets do not authorize arbitrary aliased external pointers.

Recorded by `benchmarks/record_owned_source_state_gpu.py` (`nvidia.json`, `rocm.json`).
