---
status: Informative
classification: Hardware Dtype Audit
last_updated: 2026-05-11
---

# Hardware Dtype Support Matrix

This matrix maps Tessera's cleaned dtype families onto the hardware targets
called out in the May 2026 dtype audit. It is a hardware capability view, not a
claim that every backend lowering path is implemented today. The active Tessera
compiler support remains governed by `python/tessera/compiler/graph_ir.py`,
`python/tessera/compiler/gpu_target.py`, and backend-specific legality tables.

## Legend

| Mark | Meaning |
|------|---------|
| `N` | Native fast matrix/tensor path documented for the target family. |
| `S` | Scalar/vector/storage support exists, but not a primary matrix/tensor path. |
| `P` | Packed, block-scaled, microscaled, or library-policy format; requires explicit dtype policy. |
| `E` | Emulated, converted, reference-only, or software/library fallback. |
| `-` | Not a meaningful or documented hardware/compiler target for this family. |
| `?` | Public specs are preliminary or naming is not final. |

## Target Abbreviations

| Column | Target |
|--------|--------|
| `AMD GFX12` | AMD Radeon RDNA4 / ROCm `gfx1200` and `gfx1201` class GPUs. |
| `MI355X` | AMD Instinct MI355/MI355X / CDNA 4 / ROCm `gfx950`. |
| `H100` | NVIDIA Hopper H100 / SM90. |
| `B100/B200` | NVIDIA Blackwell data-center GPUs / SM100-family planning target. |
| `R100/Rubin` | NVIDIA Rubin-family planning target. `R100` is Tessera's Rubin target shorthand. |
| `Apple M1 Max` | Apple M1 Max CPU + GPU through Accelerate/Metal/MPS-style paths. |
| `TT Blackhole` | Tenstorrent Blackhole Tensix processor. |
| `TT Wormhole` | Tenstorrent Wormhole Tensix processor. |
| `Xeon 6` | Intel Xeon 6 CPUs with AVX-512 / AMX capabilities. |
| `Zen 4/5` | AMD EPYC Zen 4 and Zen 5 CPUs. |

## Matrix

| Tessera family | Canonical Tessera names | AMD GFX12 | MI355X | H100 | B100/B200 | R100/Rubin | Apple M1 Max | TT Blackhole | TT Wormhole | Xeon 6 | Zen 4/5 | Notes |
|----------------|-------------------------|-----------|--------|------|-----------|------------|--------------|--------------|-------------|--------|---------|-------|
| FP64 | `fp64`, `f64` | S | N | N | N | N? | S | S | - | S | S | Native HPC matrix on MI355X/H100-class accelerators; CPU scalar/vector support is standard, but not a Tessera tensor-core path. |
| FP32 | `fp32`, `f32` | N | N | S | S | S? | S | S | S | S | S | FP32 is the portable reference dtype. NVIDIA Blackwell/Rubin public tables list FP32 under CUDA-core precision; FP32 matrix may be emulated through lower precision. |
| TF32 | `tf32` | - | - | N | N | N? | - | N | N | - | - | Tensor math policy, not storage. Tenstorrent exposes TF32/TensorFloat-style formats; AMD ROCm names `xf32` in rocWMMA, but GFX12/CDNA4 target docs do not make it a Tessera storage dtype. |
| FP16 | `fp16`, `f16` | N | N | N | N | N? | N | N | N | S/N | S | Xeon 6 has AVX-512 FP16 and AMX FP16 on P-core parts; Zen supports AVX-512 FP16-class vector paths depending on generation/model/toolchain. |
| BF16 | `bf16` | N | N | N | N | N? | E/S | N | N | N | S/E | Xeon 6 AMX targets BF16 directly. Apple BF16 depends on OS/Metal exposure and is not an M1 Max-era universal fast matrix contract. Zen BF16 usually rides AVX-512/library conversion rather than AMX-style tiles. |
| FP8 | `fp8_e4m3`, `fp8_e5m2` | N | N | N | N | N? | - | N/P | N/P | E/S | - | AMD MI355X supports OCP FP8; GFX12 rocWMMA supports `f8`/`bf8`. Tenstorrent FP8/BFP8 naming is not identical to OCP FP8. oneDNN exposes functional f8 types on AVX-512 FP16 systems, but not a general CPU tensor instruction. |
| FP6 | `fp6_e2m3`, `fp6_e3m2` | - | P | - | N | N? | - | - | - | - | - | NVIDIA Blackwell/Rubin list FP8/FP6 Tensor Core support. AMD MI355X supports microscaling MXFP6, which should map to an MX policy rather than raw Tessera `fp6_*` storage. |
| FP4 | `fp4_e2m1` | - | P | - | N | N? | - | P | P | E | - | NVIDIA Blackwell/Rubin use FP4/NVFP4 paths. AMD MI355X exposes MXFP4. Tenstorrent supports BLOCKFP4/BFP4-style formats, not OCP/NVIDIA FP4 semantics. |
| NVFP4 | `nvfp4` | - | - | - | N | N? | - | - | - | - | - | NVIDIA-specific block-scaled FP4 policy. Keep separate from AMD MXFP4 and Tenstorrent BFP4. |
| INT64 | `int64`, `i64` | S | S | S | S | S? | S | S | - | S | S | Host ABI, indexing, and scalar/vector integer support. Not a tensor-core LLM dtype. |
| INT32 | `int32`, `i32` | S/N | S/N | S | S | S? | S | S | S | S | S | Common accumulator/index dtype. Tenstorrent lists INT32 output-only on Tensix. |
| INT16 | `int16`, `i16` | S | S | S | S | S? | S | S | - | S | S | Storage/vector dtype on most targets; not usually a matrix acceleration format. |
| INT8 | `int8`, `i8` | N | N | N | N | N? | S/E | N | N | N | N | Matrix INT8 is common on datacenter GPUs, Tenstorrent Tensix, and CPU AI paths through AMX/VNNI-style instructions. |
| UINT8 | `uint8`, `u8` | S | S | S | S | S? | S | S | S | S | S | Byte/token/image storage. Tessera still needs explicit Graph IR unsigned mappings. |
| UINT16/32/64 | `uint16`, `uint32`, `uint64` | S | S | S | S | S? | S | S | - | S | S | General scalar/storage support on host/device APIs; not a primary tensor-core dtype family. |
| INT4 | `int4`, `i4` | - | P | P/E | P/E | P/E? | P/E | P | P | E | E | Hardware may support packed low-bit quantization through libraries or historical instruction forms, but Tessera has no first-class direct int4 IR/storage policy today. |
| MX formats | proposed `mxfp8`, `mxfp6`, `mxfp4` | - | N/P | - | - | - | - | - | - | - | - | MI355X is the primary target in this list with documented MXFP8/MXFP6/MXFP4 throughput. These need explicit Tessera canonical names and block-scale metadata. |
| Tenstorrent BFP | proposed `bfp8_b`, `bfp4_b`, `blockfp*` | - | - | - | - | - | - | N/P | N/P | - | - | Tenstorrent BLOCKFP/BFP formats are block-floating formats with shared exponent policy and should not be folded into OCP FP8, AMD MXFP, or NVIDIA NVFP4. |
| BOOL | `bool` | S | S | S | S | S? | S | S | S | S | S | Predicate/mask storage; backend lowering should treat it separately from numeric tensor compute. |

## Tessera Policy Implications

| Gap | Recommended compiler policy |
|-----|-----------------------------|
| Unsigned integer dtypes | Add explicit Graph IR mappings for `uint8`, `uint16`, `uint32`, and `uint64`; gate acceleration separately from storage legality. |
| `tf32` | Normalize as `math_mode="tf32"` on `fp32` tensors, not as a storage dtype. |
| Direct `int4` | Add a first-class packed storage policy before advertising direct target support. Keep current int8-container quantization status explicit. |
| AMD MX formats | Add canonical names `mxfp8`, `mxfp6`, and `mxfp4` with block-scale metadata and ROCm/CDNA4 target gates. |
| Tenstorrent BFP formats | Add separate `bfp*`/`blockfp*` policy names if Metalium lowering needs native Tensix formats. Do not alias them to OCP FP8/FP4. |
| Apple CPU/GPU | Keep Apple M1 Max focused on `fp32`/`fp16`/integer storage plus reference or framework BF16 where available; do not claim FP8/FP4 native support. |

## Evidence Sources

- [NVIDIA H100 product specs](https://www.nvidia.com/en-us/data-center/h100/)
  list FP64, FP64 Tensor Core, FP32, TF32 Tensor Core, BF16 Tensor Core, FP16
  Tensor Core, FP8 Tensor Core, and INT8 Tensor Core performance.
- [NVIDIA Tensor Core public specs](https://www.nvidia.com/en-us/data-center/tensor-cores/)
  list Blackwell and Rubin supported Tensor Core precisions as `NVFP4`, `FP64`,
  `TF32`, `BF16`, `FP16`, `FP8/FP6`, and `INT8`, with Rubin specifications
  marked preliminary.
- [AMD MI355X product specs](https://www.amd.com/en/products/accelerators/instinct/mi350/mi355x.html)
  list CDNA 4, OCP FP8, FP16, BF16, INT8, FP32, FP64, and MXFP8/MXFP6/MXFP4 peak
  matrix performance.
- [ROCm rocWMMA docs](https://rocm.docs.amd.com/projects/rocWMMA/en/develop/api-reference/api-reference-guide.html)
  list GFX12 support and matrix data types including `f8`/`bf8`, `f16`, `bf16`,
  `i8`, `i32`, `xf32`, and `f64` combinations.
- [Tenstorrent Blackhole specs](https://docs.tenstorrent.com/aibs/blackhole/specifications.html)
  and [Wormhole specs](https://docs.tenstorrent.com/docs-test/core/latest/aibs/wormhole/specifications.html)
  list FP8, FP16, BF16, FP32 output-only, BLOCKFP2/BLOCKFP4/BLOCKFP8 or
  BFP2/BFP4/BFP8, INT8, INT32 output-only, UINT8, TF32, and vector formats.
- [Intel Xeon 6 docs](https://www.intel.com/content/www/us/en/support/articles/000098612/processors/intel-xeon-processors.html)
  list AMX acceleration for INT8, BF16, and FP16. [oneDNN data type docs](https://www.intel.com/content/www/us/en/docs/onednn/developer-guide-reference/2025-2/data-types-001.html)
  document CPU library dtype coverage for f32, s8/u8, bf16, f16, f8, and f4
  datatypes with ISA-specific requirements.
- AMD EPYC 9005 launch and architecture material confirms Zen 5 server CPUs with
  AVX-512 and a full 512-bit data path. Zen 4 EPYC CPUs provide AVX-512-class
  vector support but not an AMX-style tile matrix path.
- Apple [Metal tensor data type docs](https://developer.apple.com/documentation/metal/mtltensordatatype),
  [MPS data type docs](https://developer.apple.com/documentation/metalperformanceshaders/mpsdatatype),
  and [Accelerate docs](https://developer.apple.com/documentation/accelerate)
  expose fp32/fp64 CPU math, Metal half/float and integer types, and
  BF16/BFloat API exposure on newer Metal stacks; M1 Max should be treated
  conservatively for native BF16/FP8/FP4 acceleration.
