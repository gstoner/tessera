# Apple Silicon Deep Learning Datatypes

> **Status:** Reference note. Apple does not publish a raw per-chip Neural Engine datatype ISA table. This document tracks the datatypes Apple exposes to developers through public frameworks such as Core ML, Metal, Metal Performance Shaders, Accelerate/BNNS, and MLX.

This table is intended as a practical backend-planning reference for Tessera. It distinguishes the visible software contract from likely hardware internals, especially for the Apple Neural Engine.

## Summary

| Apple generation | CPU exposed DL datatypes | GPU exposed DL datatypes | NPU / Neural Engine exposed datatypes | Important generation notes |
| --- | --- | --- | --- | --- |
| M1 | FP32, FP16, BF16 through ML frameworks, INT8/UINT8, INT16/INT32, FP64 CPU-only | FP32, FP16, BF16 through modern ML/Metal stacks, INT8/UINT8, INT4/UINT4 as tensor or quantized storage in newer APIs | Core ML primarily exposes FP16 and quantized INT8-style inference. Exact ANE internal formats are not public. | 16-core Neural Engine, 11 TOPS. CPU, GPU, and Neural Engine are usable through Core ML scheduling. |
| M2 | Same exposed set as M1 | Same exposed set as M1 | Same Core ML exposure: FP16 and quantized INT8-style inference | 16-core Neural Engine, 15.8 TOPS. Higher memory bandwidth helps larger models. |
| M3 | Same exposed set as M1/M2 | Same, plus newer GPU architecture features. FP32, FP16, BF16, and integer quantized paths are available through Metal, MPS, and MLX. | Same Core ML exposure | Neural Engine is described by Apple as up to 60% faster than M1. M3 GPU adds Dynamic Caching, hardware ray tracing, and mesh shading. |
| M4 | Same exposed set | Same exposed set. Metal, MPS, and MLX remain the main route for FP16, BF16, and quantized GPU workloads. | Same Core ML exposure | 16-core Neural Engine rated up to 38 TOPS. Apple also calls out next-generation ML accelerators in the CPU. |
| M5 | Same exposed set | FP32, FP16, BF16, INT8/UINT8, INT4/UINT4 tensor and quantized paths. M5 also exposes GPU Neural Accelerators through Metal 4 Tensor APIs and MLX. | Same Core ML Neural Engine exposure. Apple describes a faster 16-core Neural Engine, but no raw NPU datatype table is public. | Major change: every GPU core has a Neural Accelerator for AI matrix work. Apple MLX examples explicitly benchmark BF16, 4-bit quantized, and MXFP4 model workloads on M5. |

## Datatype Matrix

| Datatype | CPU | GPU | Apple NPU / Neural Engine |
| --- | --- | --- | --- |
| FP64 | Yes. MLX documents FP64 as CPU-only. | Not exposed for MLX GPU execution. | No public ANE exposure. |
| FP32 | Yes | Yes | Core ML may accept FP32 models, but ANE placement is graph- and runtime-dependent. |
| FP16 | Yes | Yes. Preferred for many Apple Silicon deep learning workloads. | Yes, commonly used through Core ML execution. |
| BF16 | Yes through MLX and framework-level APIs. | Yes through MLX, Metal, and MPS API surfaces. | Not publicly documented as a raw ANE datatype. |
| INT8 / UINT8 | Yes | Yes | Yes through Core ML quantized inference paths. |
| INT4 / UINT4 | Mainly quantized storage or weights. | Yes as Metal tensor datatypes and quantized ML paths. | Not directly documented. Core ML may use compressed weights depending on model and operation support. |
| FP8 | Not generally exposed as a standard Apple deep learning datatype. | Not in the public Metal tensor datatype list. | Not publicly exposed. |
| MXFP4 | MLX model/workload support appears in M5 examples. | Used in Apple MLX M5 benchmark examples. | Not a public ANE datatype claim. |

## Backend Implications

For Tessera's Apple targets, the conservative lowering strategy is:

1. Treat `f32`, `f16`, and `bf16` as the main floating-point GPU-visible datatypes.
2. Treat `f64` as CPU-only unless a specific runtime path proves otherwise.
3. Model `int8`, `uint8`, `int4`, and `uint4` primarily as inference and quantized-weight datatypes.
4. Keep Neural Engine lowering behind Core ML or framework-mediated execution rather than assuming direct ANE instruction or buffer control.
5. For M5 and later, consider a separate GPU TensorOps / Neural Accelerator capability bit instead of folding that behavior into the generic Apple GPU profile.

## Notes

- "Exposed" means available through public Apple APIs, not necessarily executed natively by every hardware block for every operation.
- Core ML can schedule work across CPU, GPU, and Neural Engine. The final placement depends on model structure, operation support, precision, OS version, and device.
- Metal and MPS expose tensor datatypes that are useful for GPU-backed machine learning. Availability of a datatype does not imply every operation has a fast specialized kernel.
- M5 is the first M-series generation where Apple publicly describes GPU Neural Accelerators in each GPU core and direct developer access through Metal 4 Tensor APIs.

## Sources

- Apple Developer Documentation, Core ML: CPU, GPU, and Neural Engine compute devices and framework scheduling.
- Apple Developer Documentation, Metal `MTLTensorDataType`: `float16`, `float32`, `bfloat16`, `int8`, `uint8`, `int4`, `uint4`, and related tensor datatypes.
- Apple Developer Documentation, Metal Performance Shaders `MPSDataType`: `float16`, `bFloat16`, `int8`, and related datatypes.
- Apple MLX documentation, Data Types: `float16`, `bfloat16`, `float32`, `float64`, integer datatypes, and FP64 CPU-only behavior.
- Apple Newsroom, "Apple unleashes M1": M1 16-core Neural Engine and 11 trillion operations per second.
- Apple Newsroom, "Apple unveils M2": M2 16-core Neural Engine and 15.8 trillion operations per second.
- Apple Newsroom, "Apple unveils M3, M3 Pro, and M3 Max": M3 GPU architecture and Neural Engine speedup.
- Apple Newsroom, "Apple introduces M4 chip": M4 16-core Neural Engine and 38 trillion operations per second.
- Apple Newsroom, "Apple unleashes M5": M5 GPU Neural Accelerators, improved 16-core Neural Engine, and 153 GB/s unified memory bandwidth.
- Apple Machine Learning Research, "Exploring LLMs with MLX and the Neural Accelerators in the M5 GPU": M5 MLX benchmarks with BF16, 4-bit quantization, and MXFP4 workloads.
