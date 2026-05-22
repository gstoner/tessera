# CPU Backend (Testing-Friendly)

This backend emits C for **native CPU execution** with selectable implementations:

- `naive`: triple loops (portable, zero deps).
- `openmp`: naive loops with `#pragma omp parallel for` on the outer `i` loop.
- `blas`: delegates matmul to `cblas_sgemm` (OpenBLAS/MKL/Accelerate).
- `avx2`: includes a simple AVX2 micro-kernel for inner-product accumulation.

## Build flags & deps

- `openmp`: requires OpenMP support (e.g., `-fopenmp` with GCC/Clang; MSVC: `/openmp`).
- `blas`: link a BLAS: for OpenBLAS `-lopenblas`; MKL or Accelerate otherwise. Adjust `Makefile` vars.
- `avx2`: compile with `-mavx2 -mfma` or equivalent; falls back to naive if not available.

## CLI

```
python -m tilec.driver examples/matmul.tss --backend cpu --impl naive   --out build/mm_naive
python -m tilec.driver examples/matmul.tss --backend cpu --impl openmp  --out build/mm_omp
python -m tilec.driver examples/matmul.tss --backend cpu --impl blas    --out build/mm_blas
python -m tilec.driver examples/matmul.tss --backend cpu --impl avx2    --out build/mm_avx2
make -C build/mm_avx2 && ./build/mm_avx2/mm
```
