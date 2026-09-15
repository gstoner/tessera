# Matched quantized MPP comparison — M1 Max, 2026-09-15

Owner IR-NATIVE-FOUNDATION-1 / APPLE-METAL41-20260914.

Fresh runtime from local base c8ea88fe: 99 accumulation/layout tests pass.
Each low-precision/FP16 pair receives identical quantized operand values;
float64 references are formed from those values. Incorrect or nonfinite outputs
fail before timing. The reported gamma-K bound isolates fp32 accumulation from
input quantization; passing it is not a universal accumulator-width proof.

Three separate processes, 30 timed repetitions per route after three warmups.
At 2048 cubed, low-precision/FP16 throughput ratios across the three processes:
E4M3 0.772–0.773; E5M2 0.897–0.898; E2M1 0.775–0.776.
Packed input storage is 50% of FP16 for FP8 and 25% for FP4.
256 and 1024 cases show substantial variation and are not stable speed claims.

Device intervals use the existing Metal4 timestamp heap. Host wall time and
packing-plus-reference-conversion time are separate. Packing includes creation
of the matched fp16 reference arrays, so it is an upper-cost characterization,
not isolated pack-kernel timing. Route timings are sequential, not randomized;
this remains characterization, not selector-grade evidence or promotion.
No inference about dedicated hardware follows from throughput ratios alone.

Reproduce with a freshly compiled runtime:

```sh
PYTHONPATH=python:. TESSERA_APPLE_GPU_RUNTIME_LIB=/path/to/runtime.dylib python3 benchmarks/apple_gpu/compare_quantized_matmul.py   --shapes 256,1024,2048 --reps 30 --out /tmp/matched.json
```

The existing matrix tests cover padded row strides, nonzero origins, partial
output tiles, cancellation, small accumulation increments and long reductions.
The existing FP4 nonzero-origin workaround remains OS-dependent and is guarded
by those tests. No runtime implementation was changed in this follow-through.
