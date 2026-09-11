# Incremental heap and traced SSD — 2026-09-11

Correctness only: Super-Bear RTX 5070 SM120 and Princess-Luna Radeon 8060S
gfx1151, both WSL. Compiler and recorder hashes and native binding identities
are retained separately. No timing, counter or promotion claim.

`record_incremental_heap_trace.py` verifies an explicitly trusted deque extractor
with a self-cycle, then incremental collection of four byte slots. An unreachable
cycle spans two sweep batches. Both members retire logically before slot reuse;
state lifecycle is 0 free, 1 live, 2 retired. A reclaimed slot is reused with a
new generation and a root is published between batches. Readers and writers
remain excluded during each batch; this is not simultaneous sweeping.

The same recorder traces two resident SSD calls through public `vjp`, with nine
distinct public inputs. It compares the composed value and every input gradient
to an independent float64 NumPy recurrence / central finite-difference oracle.
Scoped GPU readers supply the intermediate values and cotangents. Composition
is host orchestration of replay-bound packages, not a fused or canonical
whole-program IR. Context sync
is forbidden on the bound SSD path, and whole-program retirement completes
before module release. Device-to-host reads explicitly synchronize the borrowed view stream first;
they are for validation, not an overlap
measurement. An observation deadline does not cancel driver calls.

Separately, `test_attention_loop_idiom.py::test_full_shape_additive_bias_native_binding`
passed on CUDA with finite bias, irregular negative-infinity masks, and
pre-launch refusal of fully masked rows and NaNs. This is one small full-shape
case, not Boolean/broadcast-mask support or performance evidence.

Reproduce with the owning backend environment and:

```
python benchmarks/record_incremental_heap_trace.py --backend nvidia --compiler <tool> --output <packet>
```

Use `--backend rocm` on Princess-Luna. The CUDA attention check also requires
`TESSERA_TEST_RAISED_ATTENTION=1`, `TESSERA_OPT`, `TESSERA_NVIDIA_OPT`, and
`TESSERA_NVIDIA_PTX_LAUNCH_LIB` pointing to the owning tools/runtime.

Remaining: arbitrary extension discovery, concurrent sweep publication/read
barriers, shared-input gradient accumulation, nonlinear/mixed traced families,
control flow, general masks, and exact-device Apple/x86 integration.
