# Source indices, exception generations and checked GPU VJP

Owners: W4-PRODUCT-1 / AD-RESIDUAL-EVAL-1 / W2.4a.
Sync key: SOURCE-GENERATION-AD-2026-09-09.

Both owning hosts independently pass twelve cases: CUDA RTX 5070 / SM120 on
Super-Bear and ROCm gfx1151 on Princess-Luna. Each packet fingerprints its native
compiler and the measured frontend, public-result binding, AD pass, GPU pass and
recorder. Source fingerprints match this tree; evidence does not transfer to
Apple or other GPU architectures. No performance claim is made.

Cases cover positive/negative completion of a source cube and a bounded loop
with a nested cause/context, successful numeric VJP, failure without derivative
exposure, six runtime slice-bound/step combinations, and zero/negative runtime
step refusal. Slice cases include empty results and int64 extrema. One compiled
slice artifact serves every bound/step combination. VJP captures private device
input snapshots and uses only the checked forward's residuals; scalar i8 saved
predicates retain their storage type and logical rank zero.

The host WSL regression run passed 531 tests, with 33 skipped (source/tracer,
public frames, source state, dtype, diagnostics and pass metadata). Audit and
governance tests passed 25 cases. Touched-file Ruff and compiler-package mypy
(305 files) passed. Both native compilers rebuilt; Super-Bear's uses assertions-
enabled LLVM 23. No full unit suite was run.

The implementation is bounded: static ranked source roots, one-element int64
tensor bounds and positive runtime strides; at most 32 exception payload slots
across expanded loop generations; synchronous single-input exception-aware GPU
VJP, without projected aliases/object fields. Native CPU DPS output shape is
caller-provided. Automatic dynamic CPU JIT allocation, Python integer/index
protocols, negative runtime steps, nested dynamic views, arbitrary exception
objects across generations, asynchronous VJP and full CPython frames remain open.
Host failure polling no longer grows traceback chains; native source notes are
not CPython execution frames. Block AttnRes promotion remains unchanged.
