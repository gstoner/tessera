"""x86 GEMM-family lane — batched_gemm / linear_general / qkv_projection /
factorized_matmul / einsum, all on the AVX-512 f32 GEMM microkernel
(tessera_x86_avx512_gemm_f32) with reshape/batch/einsum in Python. The CPU
analog of the ROCm WMMA matmul-family lane.

Reachable through `runtime.launch()` via
`compiler_path="x86_matmul_family_compiled"`. f32; validated vs numpy at a
K-scaled tolerance (rtol 1e-3).

Skip-clean: libtessera_x86_elementwise.so absent.
"""

from __future__ import annotations

import numpy as np
import pytest


def _x86_or_skip():
    from tessera import runtime as rt
    if not rt._x86_elementwise_available():
        pytest.skip("libtessera_x86_elementwise.so not built/loadable")
    return rt


def _artifact(rt, op_name, operands, kwargs=None):
    return rt.RuntimeArtifact(metadata={
        "target": "x86", "compiler_path": "x86_matmul_family_compiled",
        "executable": True, "execution_kind": "native_cpu",
        "arg_names": list(operands), "output_name": "o",
        "ops": [{"op_name": op_name, "result": "o", "operands": list(operands),
                 "kwargs": kwargs or {}}],
    })


_TOL = dict(atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("shape", [(2, 4, 8, 16), (3, 5, 7), (8, 8)])
def test_batched_gemm_matches_numpy(shape):
    rt = _x86_or_skip()
    rng = np.random.default_rng(11 + len(shape))
    *batch, m, k = shape
    a = rng.standard_normal((*batch, m, k)).astype(np.float32)
    b = rng.standard_normal((*batch, k, m)).astype(np.float32)
    res = rt.launch(_artifact(rt, "tessera.batched_gemm", ("a", "b")), (a, b))
    assert res["ok"] is True, res.get("reason")
    assert res["compiler_path"] == "x86_matmul_family_compiled"
    np.testing.assert_allclose(np.asarray(res["output"]).astype(np.float32),
                               (a @ b).astype(np.float32), **_TOL)


def test_batched_gemm_shared_rank2_rhs():
    rt = _x86_or_skip()
    rng = np.random.default_rng(3)
    a = rng.standard_normal((4, 6, 16)).astype(np.float32)
    b = rng.standard_normal((16, 10)).astype(np.float32)
    res = rt.launch(_artifact(rt, "tessera.batched_gemm", ("a", "b")), (a, b))
    assert res["ok"] is True, res.get("reason")
    np.testing.assert_allclose(np.asarray(res["output"]).astype(np.float32),
                               (a @ b).astype(np.float32), **_TOL)


@pytest.mark.parametrize("ash,bsh", [
    ((2, 1, 5, 8), (1, 3, 8, 6)),    # mutual broadcast -> (2, 3, 5, 6)
    ((1, 4, 8), (3, 8, 6)),          # broadcast leading 1
    ((3, 5, 8), (8, 6)),             # rank-2 rhs shared across batch
    ((5, 8), (2, 8, 6)),             # rank-2 lhs shared across batch
])
def test_batched_gemm_broadcast(ash, bsh):
    rt = _x86_or_skip()
    rng = np.random.default_rng(hash((ash, bsh)) % 2**31)
    a = rng.standard_normal(ash).astype(np.float32)
    b = rng.standard_normal(bsh).astype(np.float32)
    res = rt.launch(_artifact(rt, "tessera.batched_gemm", ("a", "b")), (a, b))
    assert res["ok"] is True, res.get("reason")
    np.testing.assert_allclose(np.asarray(res["output"]).astype(np.float32),
                               (a @ b).astype(np.float32), **_TOL)


def test_gemm_rejects_non_f32():
    rt = _x86_or_skip()
    a = np.zeros((4, 8), np.float64)
    b = np.zeros((8, 4), np.float64)
    res = rt.launch(_artifact(rt, "tessera.batched_gemm", ("a", "b")), (a, b))
    assert res["ok"] is False
    assert "f32 only" in str(res.get("reason"))


def test_linear_general_with_bias():
    rt = _x86_or_skip()
    rng = np.random.default_rng(7)
    x = rng.standard_normal((4, 5, 32)).astype(np.float32)
    w = rng.standard_normal((32, 12)).astype(np.float32)
    bias = rng.standard_normal((12,)).astype(np.float32)
    res = rt.launch(_artifact(rt, "tessera.linear_general", ("x", "w", "b")),
                    (x, w, bias))
    assert res["ok"] is True, res.get("reason")
    np.testing.assert_allclose(np.asarray(res["output"]).astype(np.float32),
                               (x @ w + bias).astype(np.float32), **_TOL)


def test_qkv_projection():
    rt = _x86_or_skip()
    rng = np.random.default_rng(9)
    x = rng.standard_normal((2, 7, 16)).astype(np.float32)
    w = rng.standard_normal((16, 48)).astype(np.float32)   # 3*16
    res = rt.launch(_artifact(rt, "tessera.qkv_projection", ("x", "w")), (x, w))
    assert res["ok"] is True, res.get("reason")
    np.testing.assert_allclose(np.asarray(res["output"]).astype(np.float32),
                               (x @ w).astype(np.float32), **_TOL)


@pytest.mark.parametrize("spec,ls,rs", [
    ("bij,bjk->bik", (3, 4, 5), (3, 5, 6)),      # batched matmul
    ("ij,jk->ik", (8, 16), (16, 10)),            # plain matmul
    ("bhid,bhjd->bhij", (2, 3, 5, 7), (2, 3, 6, 7)),  # attention scores
])
def test_einsum_single_contraction(spec, ls, rs):
    rt = _x86_or_skip()
    rng = np.random.default_rng(hash(spec) % 2**31)
    a = rng.standard_normal(ls).astype(np.float32)
    b = rng.standard_normal(rs).astype(np.float32)
    res = rt.launch(_artifact(rt, "tessera.einsum", ("a", "b"),
                              {"spec": spec}), (a, b))
    assert res["ok"] is True, res.get("reason")
    np.testing.assert_allclose(np.asarray(res["output"]).astype(np.float32),
                               np.einsum(spec, a, b).astype(np.float32), **_TOL)


def test_factorized_matmul_rank_truncation():
    rt = _x86_or_skip()
    rng = np.random.default_rng(13)
    a = rng.standard_normal((12, 20)).astype(np.float32)
    b = rng.standard_normal((20, 16)).astype(np.float32)
    rank = 4
    res = rt.launch(_artifact(rt, "tessera.factorized_matmul", ("a", "b"),
                              {"rank": rank}), (a, b))
    assert res["ok"] is True, res.get("reason")
    full = (a @ b).astype(np.float32)
    u, s, vh = np.linalg.svd(full, full_matrices=False)
    ref = (u[:, :rank] * s[:rank]) @ vh[:rank, :]
    np.testing.assert_allclose(np.asarray(res["output"]).astype(np.float32),
                               ref.astype(np.float32), atol=1e-2, rtol=1e-2)


def test_einsum_multi_contraction_rejected():
    rt = _x86_or_skip()
    a = np.zeros((4, 4, 4), np.float32)
    with pytest.raises(ValueError, match="single contraction"):
        rt._execute_x86_compiled_matmul_family(
            _artifact(rt, "tessera.einsum", ("a", "b"),
                      {"spec": "ijk,ijk->i"}), (a, a))


def test_matmul_family_unknown_op_rejected():
    from tessera import runtime as rt
    a = np.zeros((4, 8), np.float32)
    with pytest.raises(ValueError, match="x86_matmul_family_compiled executor"):
        rt._execute_x86_compiled_matmul_family(
            _artifact(rt, "tessera.softmax", ("a",)), (a,))


def _at_offset(src: np.ndarray, offset: int) -> np.ndarray:
    """A C-contiguous copy of ``src`` whose data sits ``offset`` bytes past a
    64-byte boundary (numpy itself guarantees only 16)."""
    raw = np.empty(src.nbytes + 128, dtype=np.uint8)
    start = (-raw.ctypes.data) % 64 + offset
    view = raw[start:start + src.nbytes].view(src.dtype).reshape(src.shape)
    view[...] = src
    assert view.ctypes.data % 64 == offset
    return view


@pytest.mark.parametrize("m,n,k", [
    (1, 250, 33),     # M == 1: the direct (unpacked) path, tail strip
    (7, 129, 17),     # one full 8-strip panel + a 1-wide tail panel
    (64, 256, 64),
    (33, 100, 3),     # 7-strip block, last strip 4 wide
    (3, 130, 1100),   # K blocks of 512: two full + one partial (C carries the sum)
])
def test_gemm_f32_result_independent_of_b_alignment(m, n, k):
    """X86-GEMM-ALIGN-1: the kernel packs B into its own 64-byte-aligned panel,
    so the caller's B%64 must not change a single bit of the result."""
    import ctypes

    rt = _x86_or_skip()
    lib = rt._load_x86_elementwise()
    rng = np.random.default_rng(m * 1000 + n + k)
    a = rng.standard_normal((m, k)).astype(np.float32)
    b = rng.standard_normal((k, n)).astype(np.float32)
    cf = ctypes.POINTER(ctypes.c_float)
    outs = []
    for offset in (0, 4, 16, 32, 48, 60):
        bb = _at_offset(b, offset)
        out = np.full((m, n), np.nan, dtype=np.float32)
        lib.tessera_x86_avx512_gemm_f32(
            a.ctypes.data_as(cf), bb.ctypes.data_as(cf), ctypes.c_int64(m),
            ctypes.c_int64(n), ctypes.c_int64(k), out.ctypes.data_as(cf))
        outs.append(out)
    for out in outs[1:]:
        assert np.array_equal(out.view(np.uint32), outs[0].view(np.uint32))
    np.testing.assert_allclose(outs[0], a @ b, **_TOL)


@pytest.mark.parametrize("case", ["c_is_a", "c_is_b", "c_inside_a", "adjacent"])
def test_gemm_f32_overlapping_output_equals_disjoint_product(case):
    """X86-GEMM-ALIGN-1 review: the blocked kernel writes C while A and B are
    still read, so an overlapping C must be detected at entry and computed
    through scratch -- the result equals the product of the inputs' values at
    entry, bit for bit -- while a disjoint C takes the fast path."""
    import ctypes

    rt = _x86_or_skip()
    lib = rt._load_x86_elementwise()
    cf = ctypes.POINTER(ctypes.c_float)
    i64 = ctypes.c_int64
    overlap = lib.tessera_x86_avx512_gemm_f32_operands_overlap
    overlap.argtypes = [cf, cf, i64, i64, i64, cf]
    overlap.restype = ctypes.c_int
    m, n, k = {"c_is_a": (24, 40, 40), "c_is_b": (40, 24, 40),
               "c_inside_a": (16, 32, 64), "adjacent": (16, 32, 64)}[case]
    a_off, b_off, c_off = {
        "c_is_a": (0, 4096, 0),
        "c_is_b": (0, 4096, 4096),
        "c_inside_a": (0, 4096, 300),
        "adjacent": (0, m * k + m * n, m * k),  # C between A and B, touching both
    }[case]
    arena = np.random.default_rng(len(case) * 101).standard_normal(16384).astype(np.float32)
    a0 = arena[a_off:a_off + m * k].copy().reshape(m, k)
    b0 = arena[b_off:b_off + k * n].copy().reshape(k, n)
    want = np.zeros((m, n), np.float32)
    lib.tessera_x86_avx512_gemm_f32(a0.ctypes.data_as(cf), b0.ctypes.data_as(cf),
                                    i64(m), i64(n), i64(k), want.ctypes.data_as(cf))

    def at(off):
        return ctypes.cast(arena.ctypes.data + 4 * off, cf)

    assert overlap(at(a_off), at(b_off), m, n, k, at(c_off)) == int(case != "adjacent")
    lib.tessera_x86_avx512_gemm_f32(at(a_off), at(b_off), i64(m), i64(n), i64(k), at(c_off))
    got = arena[c_off:c_off + m * n].reshape(m, n)
    assert np.array_equal(got.view(np.uint32), want.view(np.uint32))
    np.testing.assert_allclose(got, a0 @ b0, **_TOL)


@pytest.mark.parametrize("m,n,k,packed", [
    (1, 4096, 4096, 0),    # M == 1: direct at any size
    (2, 256, 256, 0),      # M <= 4, B = 256 KiB: direct
    (4, 512, 512, 0),      # M <= 4, B = exactly 1 MiB: direct
    (4, 513, 512, 1),      # B just over 1 MiB: packed
    (5, 64, 64, 1),        # M > 4: packed
    (2, 1024, 1024, 1),    # M == 2, B = 4 MiB: packed
])
def test_gemm_f32_path_selection_follows_the_measured_rule(m, n, k, packed):
    """X86-GEMM-ALIGN-1 review: the packed path costs more than its reuse
    repays at small M, so the kernel reads B directly for M == 1, or M <= 4
    with B <= 1 MiB (crossover measured on Princess-Luna; evidence README)."""
    import ctypes

    rt = _x86_or_skip()
    fn = rt._load_x86_elementwise().tessera_x86_avx512_gemm_f32_uses_packed_path
    fn.argtypes = [ctypes.c_int64] * 3
    fn.restype = ctypes.c_int
    assert fn(m, n, k) == packed
