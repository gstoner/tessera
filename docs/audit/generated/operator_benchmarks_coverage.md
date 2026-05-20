<!-- AUTO-GENERATED — DO NOT EDIT BY HAND. -->
<!-- Regenerate via: python -m tessera.cli.operator_benchmarks_coverage -->

# Operator Benchmark Coverage

This dashboard maps the current support-table families to the seven active ``Tessera_Operator_Benchmarks`` groups. It also records families that intentionally live in a different benchmark harness.

| Opbench group | Support-table family | Representative ops | Coverage | Notes |
|---------------|----------------------|--------------------|----------|-------|
| ``matmul`` | ``loop_nest`` | ``matmul``, ``gemm`` | ``direct`` | Direct C++ opbench group and direct support-table primitive family. |
| ``conv2d`` | ``stencil`` | ``conv2d`` | ``direct`` | Direct NHWC convolution group; native target runtime remains backend-dependent. |
| ``flash_attention`` | ``attention`` | ``flash_attn`` | ``direct`` | Direct attention group for the canonical flash-attention primitive. |
| ``reduce`` | ``stable_reduction`` | ``reduce``, ``sum`` | ``grouped`` | Covers sum-style reductions; broader stable reductions are represented separately. |
| ``elementwise`` | ``elementwise`` | ``tanh``, ``add``, ``mul``, ``relu``, ``sigmoid`` | ``grouped`` | One opbench group covers multiple scalar elementwise primitives. |
| ``softmax_layernorm`` | ``stable_reduction+normalization`` | ``softmax``, ``layer_norm`` | ``grouped`` | Composite group spanning softmax and layer normalization. |
| ``transpose_gather`` | ``layout_transform`` | ``transpose``, ``gather`` | ``grouped`` | Current smoke runs transpose; gather remains mapped but not separately timed. |
| ``spectral`` | ``spectral`` | ``fft``, ``dct``, ``spectral_conv`` | ``not-applicable`` | Covered by benchmarks/spectral rather than Tessera_Operator_Benchmarks. |
| ``ga_ebm`` | ``geometric_algebra+energy_based`` | ``clifford_geo_product``, ``ebm_inner_step`` | ``not-applicable`` | Covered by benchmarks/apple_gpu/benchmark_ga_ebm.py. |
