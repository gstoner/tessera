import pathlib, textwrap

from ..ir import IRModule

def emit(ir: IRModule, out_dir: str, impl: str = "naive", exe_name: str = None):
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    f = ir.funcs[0]
    kernel_params = ", ".join(["float* "+p for p in f.params] + ["int M", "int N", "int K"])

    # Emit kernel body for matmul/add; we gate matmul by impl variant.
    def matmul_body(lhs, rhs, outv):
        if impl == "naive" or impl == "openmp":
            return f'''
            // {outv} = {lhs} @ {rhs}  (MxK) x (KxN) = (MxN)
            for (int i = 0; i < M; ++i) {{
              for (int j = 0; j < N; ++j) {{
                float acc = 0.0f;
                for (int k = 0; k < K; ++k) {{
                  acc += {lhs}[i*K + k] * {rhs}[k*N + j];
                }}
                {outv}[i*N + j] = acc;
              }}
            }}
            '''
        elif impl == "blas":
            return f'''
            // BLAS path: C = A @ B
            // Using column-major API for cblas_sgemm; we wrap with row-major helpers below.
            sgemm_row_major({outv}, {lhs}, {rhs}, M, N, K);
            '''
        elif impl == "avx2":
            return f'''
            // AVX2 path (fallback to scalar if !__AVX2__)
            mm_avx2({outv}, {lhs}, {rhs}, M, N, K);
            '''
        else:
            return "/* unknown impl */"

    body_parts = []
    for op in f.body:
        if op["op"] == "matmul":
            body_parts.append(matmul_body(op["lhs"], op["rhs"], op["out"]))
        elif op["op"] == "add":
            body_parts.append(f'''
            for (int i = 0; i < M; ++i) {{
              for (int j = 0; j < N; ++j) {{
                {op["out"]}[i*N + j] = {op["lhs"]}[i*N + j] + {op["rhs"]}[i*N + j];
              }}
            }}
            ''')

    omp_prag = "#pragma omp parallel for\n" if impl == "openmp" else ""

    helpers = r'''
    static inline float frand() { return (float)rand() / (float)RAND_MAX; }

    // --- Optional helpers for BLAS row-major wrapper
    #ifdef USE_CBLAS
    #include <cblas.h>
    static void sgemm_row_major(float* C, const float* A, const float* B, int M, int N, int K) {
      // Compute: C(MxN) = A(MxK) @ B(KxN), row-major storage
      // We implement via cblas_sgemm using row-major flag if available (CBLAS_ROW_MAJOR).
      #ifdef CBLAS_ROW_MAJOR
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    M, N, K, 1.0f, A, K, B, N, 0.0f, C, N);
      #else
        // If only column-major is available, transpose logic would be needed.
        // For brevity, we assume RowMajor is present in modern cblas.
        #error "CBLAS_ROW_MAJOR not defined; add a transpose wrapper here."
      #endif
    }
    #endif

    // --- Optional AVX2 micro-kernel
    #if defined(__AVX2__) && defined(__FMA__)
    #include <immintrin.h>
    static void mm_avx2(float* C, const float* A, const float* B, int M, int N, int K) {
      for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
          __m256 vacc = _mm256_setzero_ps();
          int k = 0;
          for (; k + 8 <= K; k += 8) {
            __m256 va = _mm256_loadu_ps(A + i*K + k);
            __m256 vb = _mm256_set_ps(B[(k+7)*N + j], B[(k+6)*N + j], B[(k+5)*N + j], B[(k+4)*N + j],
                                       B[(k+3)*N + j], B[(k+2)*N + j], B[(k+1)*N + j], B[(k+0)*N + j]);
            // multiply-add: vacc += va * vb (broadcast-per-lane pattern)
            vacc = _mm256_fmadd_ps(va, vb, vacc);
          }
          float acc = 0.0f;
          float tmp[8]; _mm256_storeu_ps(tmp, vacc);
          for (int t=0;t<8;++t) acc += tmp[t];
          for (; k < K; ++k) acc += A[i*K + k] * B[k*N + j];
          C[i*N + j] = acc;
        }
      }
    }
    #else
    static void mm_avx2(float* C, const float* A, const float* B, int M, int N, int K) {
      // Fallback: scalar
      for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j) {
          float acc = 0.0f;
          for (int k = 0; k < K; ++k) acc += A[i*K + k] * B[k*N + j];
          C[i*N + j] = acc;
        }
    }
    #endif
    '''

    kernel_src = textwrap.dedent(f'''
        // Auto-generated CPU backend ({impl})
        #include <stdio.h>
        #include <stdlib.h>
        #include <math.h>
        #include <time.h>

        {helpers}

        void kernel_{f.name}({kernel_params}) {{
            {omp_prag}for (int i_outer = 0; i_outer < 1; ++i_outer) {{
                (void)i_outer;
                {''.join(body_parts)}
            }}
        }}

        int main(int argc, char** argv) {{
          int M = getenv("M") ? atoi(getenv("M")) : 128;
          int N = getenv("N") ? atoi(getenv("N")) : 128;
          int K = getenv("K") ? atoi(getenv("K")) : 128;

          size_t szA = (size_t)M*K, szB = (size_t)K*N, szC = (size_t)M*N;
          float *A = (float*)aligned_alloc(64, szA*sizeof(float));
          float *B = (float*)aligned_alloc(64, szB*sizeof(float));
          float *C = (float*)aligned_alloc(64, szC*sizeof(float));
          if(!A||!B||!C) {{ fprintf(stderr, "alloc failed\\n"); return 2; }}
          srand(1234);
          for (size_t i=0;i<szA;++i) A[i] = frand();
          for (size_t i=0;i<szB;++i) B[i] = frand();
          for (size_t i=0;i<szC;++i) C[i] = 0.0f;

          // Map param names
          {"float* " + (f.params[0] if len(f.params)>0 else "P0") + " = A;"}
          {"float* " + (f.params[1] if len(f.params)>1 else "P1") + " = B;"}
          {"float* " + (f.params[2] if len(f.params)>2 else "P2") + " = C;"}

          kernel_{f.name}({", ".join((f.params + ["M","N","K"]))});

          double sum = 0.0;
          for (size_t i=0;i<szC;++i) sum += C[i];
          printf("OK ({impl})  M=%d N=%d K=%d  checksum=%.6f\\n", M,N,K, sum);
          free(A); free(B); free(C);
          return 0;
        }}
    ''').strip() + "\n"

    (out / f"{f.name}.c").write_text(kernel_src)

    # Makefile with variant flags
    omp_flag = "-fopenmp" if impl == "openmp" else ""
    avx_flag = "-mavx2 -mfma" if impl == "avx2" else ""
    blas_defs = " -DUSE_CBLAS" if impl == "blas" else ""
    blas_link = " -lopenblas" if impl == "blas" else ""  # adjust for MKL/Accelerate if needed

    makefile = textwrap.dedent(f'''
        CC ?= cc
        CFLAGS ?= -O3 -std=c11 {avx_flag}{blas_defs}
        LDFLAGS ?= {omp_flag}{blas_link}

        all: {f.name}

        {f.name}: {f.name}.c
        	$(CC) $(CFLAGS) -o {f.name} {f.name}.c $(LDFLAGS)

        clean:
        	rm -f {f.name}
    ''').strip() + "\n"
    (out / "Makefile").write_text(makefile)

    return str(out / f"{f.name}.c")
