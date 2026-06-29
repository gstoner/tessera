// Geometric-algebra (Clifford) bilinear-product kernel (f32) for the Tessera
// x86 backend — the GA lane (P12 of S_SERIES_GAP_CLOSURE_PLAN) backing
// geometric_product / wedge / left_contraction (and, via host composition,
// inner + rotor_sandwich) on Cl(3,0) (8 blades).
//
// A multivector product is a STRUCTURED BILINEAR FORM driven by a compile-time
// (sign, i, j -> out-blade) Cayley table: out[k] = Σ sign·a[i]·b[j] over the
// table triples. That is neither a flat elementwise map nor a dense GEMM — it
// is a sparse fixed contraction over the algebra's basis table. The three
// product variants are the same geometric-product table filtered at compile
// time: wedge keeps only disjoint-generator terms (i & j == 0); left
// contraction keeps grade(out) == grade(j) - grade(i). The tables below were
// lifted from tessera.ga.signature.Cl(3,0).product_table().
//
// Layout is BLADE-MAJOR ([8, n]): each blade plane is contiguous across the
// batch, so the contraction vectorizes cleanly — out_k[lane] += s·a_i[lane]·
// b_j[lane] runs as AVX-512 FMA over 16 batch lanes + a scalar tail. The host
// transposes [n, 8] <-> [8, n] around the call. f32. Matches the numpy GA
// reference exactly (the bilinear identities are exact in f32 for these small
// integer-sign sums).

#include <cstdint>

#if defined(__AVX512F__)
#include <immintrin.h>
#endif

namespace {

struct Triple { int k, i, j, s; };  // out[k] += s * a[i] * b[j]

// Cl(3,0) geometric product (all 64 nonzero terms).
const Triple GP[] = {
    {0,0,0,1},{1,0,1,1},{2,0,2,1},{3,0,3,1},{4,0,4,1},{5,0,5,1},{6,0,6,1},
    {7,0,7,1},{1,1,0,1},{0,1,1,1},{3,1,2,1},{2,1,3,1},{5,1,4,1},{4,1,5,1},
    {7,1,6,1},{6,1,7,1},{2,2,0,1},{3,2,1,-1},{0,2,2,1},{1,2,3,-1},{6,2,4,1},
    {7,2,5,-1},{4,2,6,1},{5,2,7,-1},{3,3,0,1},{2,3,1,-1},{1,3,2,1},{0,3,3,-1},
    {7,3,4,1},{6,3,5,-1},{5,3,6,1},{4,3,7,-1},{4,4,0,1},{5,4,1,-1},{6,4,2,-1},
    {7,4,3,1},{0,4,4,1},{1,4,5,-1},{2,4,6,-1},{3,4,7,1},{5,5,0,1},{4,5,1,-1},
    {7,5,2,-1},{6,5,3,1},{1,5,4,1},{0,5,5,-1},{3,5,6,-1},{2,5,7,1},{6,6,0,1},
    {7,6,1,1},{4,6,2,-1},{5,6,3,-1},{2,6,4,1},{3,6,5,1},{0,6,6,-1},{1,6,7,-1},
    {7,7,0,1},{6,7,1,1},{5,7,2,-1},{4,7,3,-1},{3,7,4,1},{2,7,5,1},{1,7,6,-1},
    {0,7,7,-1}};

// wedge (outer product): geometric-product terms with i & j == 0.
const Triple WEDGE[] = {
    {0,0,0,1},{1,0,1,1},{2,0,2,1},{3,0,3,1},{4,0,4,1},{5,0,5,1},{6,0,6,1},
    {7,0,7,1},{1,1,0,1},{3,1,2,1},{5,1,4,1},{7,1,6,1},{2,2,0,1},{3,2,1,-1},
    {6,2,4,1},{7,2,5,-1},{3,3,0,1},{7,3,4,1},{4,4,0,1},{5,4,1,-1},{6,4,2,-1},
    {7,4,3,1},{5,5,0,1},{7,5,2,-1},{6,6,0,1},{7,6,1,1},{7,7,0,1}};

// left contraction a ⌋ b: grade(out) == grade(j) - grade(i) >= 0.
const Triple LC[] = {
    {0,0,0,1},{1,0,1,1},{2,0,2,1},{3,0,3,1},{4,0,4,1},{5,0,5,1},{6,0,6,1},
    {7,0,7,1},{0,1,1,1},{2,1,3,1},{4,1,5,1},{6,1,7,1},{0,2,2,1},{1,2,3,-1},
    {4,2,6,1},{5,2,7,-1},{0,3,3,-1},{4,3,7,-1},{0,4,4,1},{1,4,5,-1},{2,4,6,-1},
    {3,4,7,1},{0,5,5,-1},{2,5,7,1},{0,6,6,-1},{1,6,7,-1},{0,7,7,-1}};

inline void table_for(int kind, const Triple** t, int* nt) {
    switch (kind) {
    case 1: *t = WEDGE; *nt = (int)(sizeof(WEDGE) / sizeof(Triple)); return;
    case 2: *t = LC;    *nt = (int)(sizeof(LC) / sizeof(Triple));    return;
    default: *t = GP;   *nt = (int)(sizeof(GP) / sizeof(Triple));    return;
    }
}

} // namespace

// A, B, out are blade-major [8, n] (8 contiguous planes of n batch lanes).
// kind: 0 = geometric_product, 1 = wedge, 2 = left_contraction.
extern "C" void tessera_x86_clifford_bilinear_f32(const float* A, const float* B,
                                                  int64_t n, int kind,
                                                  float* out) {
    const Triple* tbl;
    int ntri;
    table_for(kind, &tbl, &ntri);
    for (int64_t b = 0; b < 8 * n; ++b)
        out[b] = 0.0f;
    for (int t = 0; t < ntri; ++t) {
        const Triple& tr = tbl[t];
        const float* ai = A + (int64_t)tr.i * n;
        const float* bj = B + (int64_t)tr.j * n;
        float* ok = out + (int64_t)tr.k * n;
        const float s = (float)tr.s;
        int64_t lane = 0;
#if defined(__AVX512F__)
        const __m512 vs = _mm512_set1_ps(s);
        for (; lane + 16 <= n; lane += 16) {
            __m512 va = _mm512_loadu_ps(ai + lane);
            __m512 vb = _mm512_loadu_ps(bj + lane);
            __m512 vo = _mm512_loadu_ps(ok + lane);
            // ok += s * ai * bj
            vo = _mm512_fmadd_ps(_mm512_mul_ps(vs, va), vb, vo);
            _mm512_storeu_ps(ok + lane, vo);
        }
#endif
        for (; lane < n; ++lane)
            ok[lane] += s * ai[lane] * bj[lane];
    }
}
