# Signed nested source maps and asynchronous source VJP

Owners: W4-PRODUCT-1 / AD-RESIDUAL-EVAL-1 / W2.4a.
Sync key: SOURCE-NESTED-ASYNC-2026-09-09.

Recorded independently on Super-Bear (RTX 5070, SM120) and Princess-Luna
(gfx1151), using each host's compiler. Each packet fingerprints the relevant
sources and compiler and contains 19 correctness cases: signed/nested runtime
slices, zero-step refusal, synchronous/asynchronous checked VJP, loop failures
and retention of an earlier iteration's exception as an explicit cause.

Reproduce with `benchmarks/record_source_nested_async_gpu.py --backend nvidia`
or `--backend rocm`, passing `--compiler` and `--output` on the owning host.
No timing, overlap or performance promotion is claimed. Inputs must be ready
on the submitted stream; explicit close may synchronize. The source VJP binding
is single-input, without projected state aliases or object fields. Dynamic
runtime gather differentiation remains unsupported; these slice cases prove
forward execution only. Apple evidence is absent.

Separate WSL CPU tests validate Python integer indices, automatic dynamic output
allocation (including multiple/multidimensional and empty results), bounded
exception-reference retention, and native memref extent checks. These use the
CPU JIT and do not inherit GPU execution proof. Automatic host result capacity
is bounded to 1024 elements. Arbitrary custom exception objects and real CPython
frame/locals reconstruction remain open.
