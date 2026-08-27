"""Cooperative-grid CUDA Krylov solvers for dense operators.

This module owns the first NVIDIA solver package whose iteration state never
leaves device memory.  The operator is an arbitrary square row-major dense
matrix; CG additionally requires it to be SPD, while restarted GMRES admits a
general nonsingular matrix.  Dot products and norms use a deterministic
two-level reduction across a cooperative CUDA grid.

The matrix and right-hand side may use f32, f16, or bf16 storage.  Krylov
vectors and all reductions are f32.  Native low-precision *matmul* admission is
handled separately by the solver residual child and the shipped mma.sync GEMM;
this persistent matrix-vector kernel intentionally converts stored values to
f32 before multiplication so the Krylov convergence contract is unambiguous.
"""

from __future__ import annotations

import ctypes
from typing import Any

from .kernel_emitter import KernelSource, SpecPolicy
from .nvidia_cuda import _load_lib, _nvidia_cuda_compile_fn, _ptr


_ENTRY = "tessera_nvidia_dense_krylov"
_artifact: Any | None = None


def _source() -> str:
    return r'''#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cooperative_groups.h>
#include <math.h>

namespace cg = cooperative_groups;

template <typename T> __device__ float tsr_load(const T *x, long i);
template <> __device__ float tsr_load<float>(const float *x, long i) { return x[i]; }
template <> __device__ float tsr_load<__half>(const __half *x, long i) { return __half2float(x[i]); }
template <> __device__ float tsr_load<__nv_bfloat16>(const __nv_bfloat16 *x, long i) { return __bfloat162float(x[i]); }

__device__ float tsr_block_sum(float value) {
  __shared__ float lane[256];
  int t = threadIdx.x;
  lane[t] = value;
  __syncthreads();
  for (int stride = 128; stride; stride >>= 1) {
    if (t < stride) lane[t] += lane[t + stride];
    __syncthreads();
  }
  return lane[0];
}

__device__ float tsr_grid_sum(float local, float *partials, float *scalar,
                              cg::grid_group grid) {
  float block = tsr_block_sum(local);
  if (threadIdx.x == 0) partials[blockIdx.x] = block;
  grid.sync();
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    float total = 0.0f;
    // Fixed block-index order makes the reduction reproducible for a fixed
    // launch geometry; no atomics participate in the numerical result.
    for (int i = 0; i < gridDim.x; ++i) total += partials[i];
    *scalar = total;
  }
  grid.sync();
  return *scalar;
}

template <typename T>
__device__ void tsr_matvec(const T *a, const float *x, float *y, int n,
                           cg::grid_group grid) {
  long gid = (long)blockIdx.x * blockDim.x + threadIdx.x;
  long stride = (long)gridDim.x * blockDim.x;
  for (long row = gid; row < n; row += stride) {
    float sum = 0.0f;
    const T *arow = a + row * (long)n;
    for (int col = 0; col < n; ++col) sum = fmaf(tsr_load(arow, col), x[col], sum);
    y[row] = sum;
  }
  grid.sync();
}

template <typename T>
__global__ void tsr_dense_cg(
    const T *a, const T *b, float *x, float *r, float *p, float *ap,
    float *workspace, float *partials, float *scalars, int n, float tolerance,
    int max_iterations, int residual_check_interval, int *iterations,
    int *status, float *residual_norm) {
  cg::grid_group grid = cg::this_grid();
  long gid = (long)blockIdx.x * blockDim.x + threadIdx.x;
  long stride = (long)gridDim.x * blockDim.x;
  float local_b = 0.0f;
  for (long i = gid; i < n; i += stride) {
    float bi = tsr_load(b, i);
    x[i] = 0.0f; r[i] = bi; p[i] = bi; ap[i] = 0.0f; workspace[i] = 0.0f;
    local_b += bi * bi;
  }
  grid.sync();
  float rhs_sq = tsr_grid_sum(local_b, partials, &scalars[0], grid);
  if (gid == 0) {
    scalars[1] = rhs_sq; // recursive r^T r
    scalars[2] = tolerance * fmaxf(1.0f, sqrtf(rhs_sq));
    *iterations = 0; *status = sqrtf(rhs_sq) <= scalars[2] ? 1 : 0;
    *residual_norm = sqrtf(rhs_sq);
  }
  grid.sync();

  for (int iteration = 0; iteration < max_iterations && *status == 0; ++iteration) {
    tsr_matvec(a, p, ap, n, grid);
    float local_pap = 0.0f;
    for (long i = gid; i < n; i += stride) local_pap += p[i] * ap[i];
    float pap = tsr_grid_sum(local_pap, partials, &scalars[3], grid);
    if (gid == 0) {
      if (!(pap > 0.0f) || !isfinite(pap)) *status = 2;
      else scalars[4] = scalars[1] / pap; // alpha
    }
    grid.sync();
    if (*status != 0) break;

    float local_new = 0.0f;
    for (long i = gid; i < n; i += stride) {
      x[i] += scalars[4] * p[i];
      r[i] -= scalars[4] * ap[i];
      local_new += r[i] * r[i];
    }
    float rs_new = tsr_grid_sum(local_new, partials, &scalars[5], grid);
    if (gid == 0) {
      *iterations = iteration + 1;
      *residual_norm = sqrtf(rs_new);
      scalars[6] = ((*residual_norm <= scalars[2]) ||
                    ((iteration + 1) % residual_check_interval == 0)) ? 1.0f : 0.0f;
    }
    grid.sync();

    if (scalars[6] != 0.0f) {
      // A convergence decision is always based on the true residual b-Ax.
      tsr_matvec(a, x, workspace, n, grid);
      float local_true = 0.0f;
      for (long i = gid; i < n; i += stride) {
        r[i] = tsr_load(b, i) - workspace[i];
        local_true += r[i] * r[i];
      }
      float true_sq = tsr_grid_sum(local_true, partials, &scalars[7], grid);
      if (gid == 0) {
        *residual_norm = sqrtf(true_sq);
        if (*residual_norm <= scalars[2]) *status = 1;
        else scalars[5] = true_sq;
      }
      grid.sync();
      if (*status == 1) break;
      // Residual replacement prevents recursive-drift accumulation.
      for (long i = gid; i < n; i += stride) p[i] = r[i];
      if (gid == 0) scalars[1] = scalars[5];
      grid.sync();
      continue;
    }

    if (gid == 0) {
      scalars[8] = rs_new / scalars[1]; // beta
      scalars[1] = rs_new;
    }
    grid.sync();
    for (long i = gid; i < n; i += stride) p[i] = r[i] + scalars[8] * p[i];
    grid.sync();
  }
  if (gid == 0 && *status == 0) *status = 3;
  grid.sync();
  // Return A*x as the final matvec state, not a stale search-direction product.
  tsr_matvec(a, x, ap, n, grid);
  for (long i = gid; i < n; i += stride) r[i] = tsr_load(b, i) - ap[i];
}

template <typename T>
__global__ void tsr_dense_gmres(
    const T *a, const T *b, float *x, float *r, float *w, float *basis,
    float *h, float *cs, float *sn, float *g, float *y, float *partials,
    float *scalars, int n, float tolerance, int max_iterations, int restart,
    int *iterations, int *status, float *residual_norm) {
  cg::grid_group grid = cg::this_grid();
  long gid = (long)blockIdx.x * blockDim.x + threadIdx.x;
  long stride = (long)gridDim.x * blockDim.x;
  float local_b = 0.0f;
  for (long i = gid; i < n; i += stride) {
    float bi = tsr_load(b, i); x[i] = 0.0f; r[i] = bi; w[i] = 0.0f;
    local_b += bi * bi;
  }
  grid.sync();
  float rhs_sq = tsr_grid_sum(local_b, partials, &scalars[0], grid);
  if (gid == 0) {
    scalars[1] = tolerance * fmaxf(1.0f, sqrtf(rhs_sq));
    *iterations = 0; *status = sqrtf(rhs_sq) <= scalars[1] ? 1 : 0;
    *residual_norm = sqrtf(rhs_sq);
  }
  grid.sync();

  while (*iterations < max_iterations && *status == 0) {
    tsr_matvec(a, x, w, n, grid);
    float local_r = 0.0f;
    for (long i = gid; i < n; i += stride) {
      r[i] = tsr_load(b, i) - w[i]; local_r += r[i] * r[i];
    }
    float beta_sq = tsr_grid_sum(local_r, partials, &scalars[2], grid);
    if (gid == 0) {
      scalars[3] = sqrtf(beta_sq);
      *residual_norm = scalars[3];
      if (*residual_norm <= scalars[1]) *status = 1;
    }
    grid.sync();
    if (*status == 1) break;

    for (long i = gid; i < n; i += stride) basis[i] = r[i] / scalars[3];
    for (long i = gid; i < (long)(restart + 1) * restart; i += stride) h[i] = 0.0f;
    for (long i = gid; i < restart; i += stride) { cs[i] = 0.0f; sn[i] = 0.0f; y[i] = 0.0f; }
    for (long i = gid; i < restart + 1; i += stride) g[i] = 0.0f;
    if (gid == 0) { g[0] = scalars[3]; scalars[4] = 0.0f; scalars[5] = 0.0f; }
    grid.sync();

    for (int j = 0; j < restart && *iterations < max_iterations; ++j) {
      const float *vj = basis + (long)j * n;
      tsr_matvec(a, vj, w, n, grid);

      // Twice-modified Gram-Schmidt protects orthogonality on difficult
      // matrices while retaining a deterministic reduction order.
      for (int pass = 0; pass < 2; ++pass) {
        for (int i = 0; i <= j; ++i) {
          const float *vi = basis + (long)i * n;
          float local_dot = 0.0f;
          for (long k = gid; k < n; k += stride) local_dot += vi[k] * w[k];
          float dot = tsr_grid_sum(local_dot, partials, &scalars[6], grid);
          if (gid == 0) h[(long)i * restart + j] += dot;
          grid.sync();
          for (long k = gid; k < n; k += stride) w[k] -= dot * vi[k];
          grid.sync();
        }
      }
      float local_norm = 0.0f;
      for (long k = gid; k < n; k += stride) local_norm += w[k] * w[k];
      float norm_sq = tsr_grid_sum(local_norm, partials, &scalars[7], grid);
      if (gid == 0) {
        h[(long)(j + 1) * restart + j] = sqrtf(norm_sq);
        scalars[8] = h[(long)(j + 1) * restart + j] <=
                     16.0f * 1.1920928955078125e-7f * fmaxf(1.0f, scalars[3]) ? 1.0f : 0.0f;
      }
      grid.sync();
      if (scalars[8] == 0.0f) {
        float inv = 1.0f / h[(long)(j + 1) * restart + j];
        for (long k = gid; k < n; k += stride) basis[(long)(j + 1) * n + k] = w[k] * inv;
      }
      grid.sync();

      if (gid == 0) {
        for (int i = 0; i < j; ++i) {
          float a0 = h[(long)i * restart + j];
          float a1 = h[(long)(i + 1) * restart + j];
          h[(long)i * restart + j] = cs[i] * a0 + sn[i] * a1;
          h[(long)(i + 1) * restart + j] = -sn[i] * a0 + cs[i] * a1;
        }
        float a0 = h[(long)j * restart + j];
        float a1 = h[(long)(j + 1) * restart + j];
        float rho = hypotf(a0, a1);
        cs[j] = rho == 0.0f ? 1.0f : a0 / rho;
        sn[j] = rho == 0.0f ? 0.0f : a1 / rho;
        h[(long)j * restart + j] = cs[j] * a0 + sn[j] * a1;
        h[(long)(j + 1) * restart + j] = 0.0f;
        float gj = g[j];
        g[j] = cs[j] * gj;
        g[j + 1] = -sn[j] * gj;
        *residual_norm = fabsf(g[j + 1]);
        *iterations += 1;
        scalars[4] = (float)(j + 1); // inner dimension
        scalars[5] = (*residual_norm <= scalars[1] || scalars[8] != 0.0f) ? 1.0f : 0.0f;
      }
      grid.sync();
      if (scalars[5] != 0.0f) break;
    }

    if (gid == 0) {
      int inner = (int)scalars[4];
      for (int i = inner - 1; i >= 0; --i) {
        float value = g[i];
        for (int k = i + 1; k < inner; ++k) value -= h[(long)i * restart + k] * y[k];
        float diagonal = h[(long)i * restart + i];
        if (!isfinite(diagonal) || fabsf(diagonal) <= 1.0e-20f) {
          *status = 2; y[i] = 0.0f;
        } else y[i] = value / diagonal;
      }
    }
    grid.sync();
    if (*status == 2) break;
    int inner = (int)scalars[4];
    for (long k = gid; k < n; k += stride) {
      float update = 0.0f;
      for (int i = 0; i < inner; ++i) update = fmaf(basis[(long)i * n + k], y[i], update);
      x[k] += update;
    }
    grid.sync();

    // Estimated Givens residual never establishes convergence by itself.
    tsr_matvec(a, x, w, n, grid);
    float local_true = 0.0f;
    for (long k = gid; k < n; k += stride) {
      r[k] = tsr_load(b, k) - w[k]; local_true += r[k] * r[k];
    }
    float true_sq = tsr_grid_sum(local_true, partials, &scalars[9], grid);
    if (gid == 0) {
      *residual_norm = sqrtf(true_sq);
      if (*residual_norm <= scalars[1]) *status = 1;
      else if (scalars[8] != 0.0f) *status = 2;
    }
    grid.sync();
  }
  if (gid == 0 && *status == 0) *status = 3;
}

template <typename T>
int tsr_launch(const void *ha, const void *hb, float *hx, float *hr, float *haux,
               int n, int algorithm, float tolerance, int max_iterations,
               int restart, int requested_blocks, int repetitions,
               int *iterations, int *status, float *residual_norm,
               int *launched_blocks, float *elapsed_ms) {
  const int threads = 256;
  int device = 0, cooperative = 0, sms = 0, active = 0;
  if (cudaGetDevice(&device) != cudaSuccess ||
      cudaDeviceGetAttribute(&cooperative, cudaDevAttrCooperativeLaunch, device) != cudaSuccess ||
      cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device) != cudaSuccess || !cooperative)
    return 4;
  void *kernel = algorithm == 0 ? (void *)tsr_dense_cg<T> : (void *)tsr_dense_gmres<T>;
  if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active, kernel, threads, 0) != cudaSuccess)
    return 4;
  int capacity = active * sms;
  int useful = (n + threads - 1) / threads;
  int blocks = requested_blocks > 0 ? requested_blocks : useful;
  blocks = max(1, min(blocks, min(useful, capacity)));
  *launched_blocks = blocks;

  size_t matrix_bytes = (size_t)n * n * sizeof(T);
  size_t input_bytes = (size_t)n * sizeof(T);
  size_t vector_bytes = (size_t)n * sizeof(float);
  T *a = nullptr, *b = nullptr;
  float *x = nullptr, *r = nullptr, *p = nullptr, *ap = nullptr, *workspace = nullptr;
  float *basis = nullptr, *h = nullptr, *cs = nullptr, *sn = nullptr, *g = nullptr, *y = nullptr;
  float *partials = nullptr, *scalars = nullptr;
  int *di = nullptr, *ds = nullptr;
  float *dn = nullptr;
  cudaEvent_t start = nullptr, stop = nullptr;
  int rc = 3;
  if (cudaMalloc(&a, matrix_bytes) != cudaSuccess || cudaMalloc(&b, input_bytes) != cudaSuccess ||
      cudaMalloc(&x, vector_bytes) != cudaSuccess || cudaMalloc(&r, vector_bytes) != cudaSuccess ||
      cudaMalloc(&p, vector_bytes) != cudaSuccess || cudaMalloc(&ap, vector_bytes) != cudaSuccess ||
      cudaMalloc(&workspace, vector_bytes) != cudaSuccess ||
      cudaMalloc(&basis, (size_t)(restart + 1) * n * sizeof(float)) != cudaSuccess ||
      cudaMalloc(&h, (size_t)(restart + 1) * restart * sizeof(float)) != cudaSuccess ||
      cudaMalloc(&cs, (size_t)restart * sizeof(float)) != cudaSuccess ||
      cudaMalloc(&sn, (size_t)restart * sizeof(float)) != cudaSuccess ||
      cudaMalloc(&g, (size_t)(restart + 1) * sizeof(float)) != cudaSuccess ||
      cudaMalloc(&y, (size_t)restart * sizeof(float)) != cudaSuccess ||
      cudaMalloc(&partials, (size_t)blocks * sizeof(float)) != cudaSuccess ||
      cudaMalloc(&scalars, 16 * sizeof(float)) != cudaSuccess ||
      cudaMalloc(&di, sizeof(int)) != cudaSuccess || cudaMalloc(&ds, sizeof(int)) != cudaSuccess ||
      cudaMalloc(&dn, sizeof(float)) != cudaSuccess) goto done;
  if (cudaMemcpy(a, ha, matrix_bytes, cudaMemcpyHostToDevice) != cudaSuccess ||
      cudaMemcpy(b, hb, input_bytes, cudaMemcpyHostToDevice) != cudaSuccess ||
      cudaEventCreate(&start) != cudaSuccess || cudaEventCreate(&stop) != cudaSuccess) goto done;
  if (cudaEventRecord(start) != cudaSuccess) goto done;
  for (int rep = 0; rep < repetitions; ++rep) {
    if (algorithm == 0) {
      int check_interval = 8;
      void *args[] = {&a, &b, &x, &r, &p, &ap, &workspace, &partials, &scalars,
                      &n, &tolerance, &max_iterations, &check_interval, &di, &ds, &dn};
      if (cudaLaunchCooperativeKernel(kernel, blocks, threads, args, 0, nullptr) != cudaSuccess) goto done;
    } else {
      void *args[] = {&a, &b, &x, &r, &workspace, &basis, &h, &cs, &sn, &g, &y,
                      &partials, &scalars, &n, &tolerance, &max_iterations, &restart,
                      &di, &ds, &dn};
      if (cudaLaunchCooperativeKernel(kernel, blocks, threads, args, 0, nullptr) != cudaSuccess) goto done;
    }
  }
  if (cudaEventRecord(stop) != cudaSuccess || cudaEventSynchronize(stop) != cudaSuccess ||
      cudaEventElapsedTime(elapsed_ms, start, stop) != cudaSuccess) goto done;
  *elapsed_ms /= repetitions;
  if (cudaMemcpy(hx, x, vector_bytes, cudaMemcpyDeviceToHost) != cudaSuccess ||
      cudaMemcpy(hr, r, vector_bytes, cudaMemcpyDeviceToHost) != cudaSuccess ||
      cudaMemcpy(haux, algorithm == 0 ? ap : workspace, vector_bytes, cudaMemcpyDeviceToHost) != cudaSuccess ||
      cudaMemcpy(iterations, di, sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess ||
      cudaMemcpy(status, ds, sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess ||
      cudaMemcpy(residual_norm, dn, sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) goto done;
  rc = 1;
done:
  if (start) cudaEventDestroy(start); if (stop) cudaEventDestroy(stop);
  if (a) cudaFree(a); if (b) cudaFree(b); if (x) cudaFree(x); if (r) cudaFree(r);
  if (p) cudaFree(p); if (ap) cudaFree(ap); if (workspace) cudaFree(workspace);
  if (basis) cudaFree(basis); if (h) cudaFree(h); if (cs) cudaFree(cs); if (sn) cudaFree(sn);
  if (g) cudaFree(g); if (y) cudaFree(y); if (partials) cudaFree(partials);
  if (scalars) cudaFree(scalars); if (di) cudaFree(di); if (ds) cudaFree(ds); if (dn) cudaFree(dn);
  return rc;
}

extern "C" int tessera_nvidia_dense_krylov(
    const void *a, const void *b, float *x, float *r, float *aux, int n,
    int algorithm, int storage, float tolerance, int max_iterations, int restart,
    int requested_blocks, int repetitions, int *iterations, int *status,
    float *residual_norm, int *launched_blocks, float *elapsed_ms) {
  if (!a || !b || !x || !r || !aux || !iterations || !status || !residual_norm ||
      !launched_blocks || !elapsed_ms || n <= 0 || algorithm < 0 || algorithm > 1 ||
      storage < 0 || storage > 2 || !(tolerance > 0.0f) || max_iterations <= 0 ||
      restart <= 0 || restart > 32 || requested_blocks < 0 || repetitions <= 0) return 2;
  if (storage == 0) return tsr_launch<float>(a, b, x, r, aux, n, algorithm, tolerance,
      max_iterations, restart, requested_blocks, repetitions, iterations, status,
      residual_norm, launched_blocks, elapsed_ms);
  if (storage == 1) return tsr_launch<__half>(a, b, x, r, aux, n, algorithm, tolerance,
      max_iterations, restart, requested_blocks, repetitions, iterations, status,
      residual_norm, launched_blocks, elapsed_ms);
  return tsr_launch<__nv_bfloat16>(a, b, x, r, aux, n, algorithm, tolerance,
      max_iterations, restart, requested_blocks, repetitions, iterations, status,
      residual_norm, launched_blocks, elapsed_ms);
}
'''


def _storage(array: Any) -> tuple[Any, int, str]:
    import numpy as np

    value = np.ascontiguousarray(array)
    if value.dtype == np.float32:
        return value, 0, "f32"
    if value.dtype == np.float16:
        return value, 1, "f16"
    try:
        import ml_dtypes

        if value.dtype == np.dtype(ml_dtypes.bfloat16):
            return value, 2, "bf16"
    except ImportError:
        pass
    raise ValueError(f"NVIDIA dense Krylov supports f32/f16/bf16 storage; got {value.dtype}")


def _lib() -> Any:
    global _artifact
    if _artifact is None:
        _artifact = _nvidia_cuda_compile_fn(KernelSource(
            source=_source(), entry=_ENTRY, lang="cuda", spec=SpecPolicy.DYNAMIC,
            shape_key=("dense-krylov-cooperative-grid-v1",),
        ))
    return _load_lib(_artifact)


def run_dense_krylov(
    matrix: Any,
    rhs: Any,
    *,
    algorithm: str,
    tolerance: float,
    max_iterations: int,
    restart: int = 16,
    reduction_ctas: int = 0,
    repetitions: int = 1,
) -> tuple[Any, Any, Any, dict[str, Any]]:
    """Run cooperative-grid CG or restarted GMRES over a dense operator."""
    import numpy as np

    a, storage, storage_name = _storage(matrix)
    b, other, rhs_storage = _storage(rhs)
    if a.ndim != 2 or a.shape[0] != a.shape[1] or b.shape != (a.shape[0],):
        raise ValueError("NVIDIA dense Krylov requires A[n,n] and b[n]")
    if storage != other or a.dtype != b.dtype or storage_name != rhs_storage:
        raise ValueError("NVIDIA dense Krylov matrix/RHS storage must match")
    if not a.size or not np.all(np.isfinite(np.asarray(a, dtype=np.float32))):
        raise ValueError("NVIDIA dense Krylov requires a non-empty finite operator")
    if not np.all(np.isfinite(np.asarray(b, dtype=np.float32))):
        raise ValueError("NVIDIA dense Krylov requires a finite RHS")
    if algorithm not in {"cg", "gmres"}:
        raise ValueError("NVIDIA dense Krylov algorithm must be cg or gmres")
    if not np.isfinite(tolerance) or tolerance <= 0 or max_iterations <= 0:
        raise ValueError("NVIDIA dense Krylov requires positive tolerance/iterations")
    if restart <= 0 or restart > 32 or reduction_ctas < 0 or repetitions <= 0:
        raise ValueError("NVIDIA dense Krylov restart must be 1..32 and CTA/repetition counts nonnegative")

    n = a.shape[0]
    x = np.empty(n, np.float32)
    residual = np.empty(n, np.float32)
    auxiliary = np.empty(n, np.float32)
    iterations = ctypes.c_int()
    status = ctypes.c_int()
    residual_norm = ctypes.c_float()
    launched_blocks = ctypes.c_int()
    elapsed_ms = ctypes.c_float()
    fn = getattr(_lib(), _ENTRY)
    fn.restype = ctypes.c_int
    fn.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_int] * 3 + [ctypes.c_float]
    fn.argtypes += [ctypes.c_int] * 4
    fn.argtypes += [ctypes.POINTER(ctypes.c_int)] * 2
    fn.argtypes += [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_float)]
    rc = fn(
        _ptr(a), _ptr(b), _ptr(x), _ptr(residual), _ptr(auxiliary), n,
        0 if algorithm == "cg" else 1, storage, float(tolerance),
        int(max_iterations), int(restart), int(reduction_ctas), int(repetitions),
        ctypes.byref(iterations), ctypes.byref(status), ctypes.byref(residual_norm),
        ctypes.byref(launched_blocks), ctypes.byref(elapsed_ms),
    )
    if rc == 4:
        raise RuntimeError("NVIDIA dense Krylov requires cooperative-grid launch support")
    if rc != 1:
        raise RuntimeError(f"NVIDIA dense Krylov CUDA launch failed with status {rc}")
    names = {0: "running", 1: "converged", 2: "breakdown_or_non_spd", 3: "max_iterations"}
    info = {
        "algorithm": algorithm,
        "iterations": int(iterations.value),
        "status": int(status.value),
        "status_name": names.get(int(status.value), "unknown"),
        "converged": status.value == 1,
        "residual_norm": float(residual_norm.value),
        "state_residency": "single_cooperative_launch_device_resident",
        "reduction": "deterministic_multi_cta_two_level",
        "reduction_ctas": int(launched_blocks.value),
        "storage": storage_name,
        "accumulation": "f32",
        "operator": "arbitrary_dense_row_major_v1",
        "restart": int(restart),
        "device_elapsed_ms": float(elapsed_ms.value),
        "timed_repetitions": int(repetitions),
    }
    if not info["converged"]:
        raise RuntimeError(f"NVIDIA dense {algorithm} did not converge: {info}")
    return x, residual, auxiliary, info


__all__ = ["run_dense_krylov"]
