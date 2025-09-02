# Tessera x86 Backend (AVX-512 + AMX)

This is a **reference backend** for the Tessera programming model targeting Intel® x86 CPUs with **AVX‑512** and **AMX** (Advanced Matrix Extensions). It includes:

- A minimal backend registration stub (`src/backend_x86.cpp`)
- A lowering skeleton (`src/lowering_x86_mlir.cpp` placeholder) showing how Tessera IR ops could map to x86 intrinsics
- High‑performance math kernels:
  - `amx_gemm_bf16` using AMX BF16 tiles (`_tile_dpbf16ps`)
  - `avx512_gemm_bf16` using AVX‑512 BF16 (`_mm512_dpbf16_ps`) or emulation fallback
- A small runtime to **enable AMX** at runtime on Linux via `arch_prctl(ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA)` (see `src/runtime/amx_runtime.*`)
- A simple test that validates the GEMM paths (`tests/test_gemm.cpp`)

> ⚠️ AMX requires OS support. On Linux, recent kernels/glibc support `arch_prctl` with `ARCH_REQ_XCOMP_PERM` for AMX tile data permission. If this call fails, the code gracefully falls back to AVX‑512 where possible.

## Build

```bash
mkdir -p build && cd build
cmake -DTESSERA_ENABLE_AMX=ON -DTESSERA_ENABLE_AVX512=ON ..
cmake --build . -j
./test_gemm
```

If your compiler supports it, CMake will add `-mavx512*` and `-mamx-*` flags automatically.

## Integration

- The `include/tessera/x86/target.h` declares a minimal `X86Backend` interface you can adapt to your compiler driver.
- The `src/backend_x86.cpp` registers/instantiates a backend and exposes C APIs:
  - `tessera_x86_amx_available()`
  - `tessera_x86_avx512bf16_available()`
  - `tessera_x86_amx_gemm_bf16(...)`
  - `tessera_x86_avx512_gemm_bf16(...)`

## Notes

- `src/lowering_x86_mlir.cpp` is provided as a **skeleton** with comments indicating the intended rewrite patterns from Tessera IR → LLVM IR/x86 intrinsics. Plug this into your compiler’s pass pipeline.
- The provided GEMM requires `M,N` to be multiples of the tile shape (16×64 by default). The test uses padded sizes. You can extend the kernels with edge handling for production.


## New in this drop

- **INT8 GEMM**
  - AMX: `tessera_x86_amx_gemm_s8s8_s32` using `_tile_dpbssd`
  - AVX‑512 VNNI: `tessera_x86_avx512_vnni_gemm_u8s8_s32` using `_mm512_dpbusd_epi32`
- **Epilogues**
  - `tessera_x86_epilogue_bias_fp32`, `tessera_x86_epilogue_bias_gelu_fp32`
- **Edge handling**
  - AMX kernels pack partial tiles into padded stack buffers
  - AVX‑512 kernels include scalar cleanup paths for tails
