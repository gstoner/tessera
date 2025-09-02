# x86 Backend Overview

This backend targets Intel® CPUs with AVX‑512 and AMX. It contains:

- **Runtime feature detection** (CPUID + Linux `arch_prctl` AMX permission)
- **AMX tile configuration helpers**
- **BF16 GEMM** using AMX tiles and AVX‑512 BF16
- **Lowering skeleton** to guide mapping from Tessera IR to x86 intrinsics/LLVM IR

See also: `tessera_target_ir_to_x86.md`.

## Runtime feature detection

- AVX‑512 BF16: CPUID.(EAX=7,ECX=1):EAX[5] (AVX512_BF16) on some CPUs; compilers define `__AVX512BF16__`.
- AMX: CPUID.(EAX=7,ECX=0):EDX[24]=AMX_TILE, [25]=AMX_INT8, [22]=AMX_BF16.
- On Linux, request tile permission via `arch_prctl(ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA /* 18 */)`.

The runtime provides:

```c++
bool tessera_x86_amx_supported();     // hardware support bit
bool tessera_x86_amx_enable_linux();  // request permission; returns true on success
bool tessera_x86_avx512bf16_supported();
```

## GEMM tile shapes

We use a simple, performant default:

- `TC`: 16×64 FP32 accumulators
- `A`: 16×64 (BF16)
- `B`: 64×16 (BF16)

You can adjust shapes by editing the tile config builder in `src/runtime/amx_runtime.cpp`.
