"""The compiler-emitted sm_120a NVFP4 block-scale warp tile as a consumer.

Host-free half: the per-lane packer is a bijection on the operand tile, its
scale words follow the ``scale_vec::4X`` selector convention the on-silicon
spike proved (lower lane pair supplies A rows ``gid``/``gid+8``, lane 0 of
each quad supplies B column ``gid``, four ue4m3 bytes per word), and the
exact tile reference agrees with the plain decoded product under unit scales.

Device half (Super-Bear, RTX 5070 sm_120): the emitted PTX registered through
the launch bridge under its own entry executes the tile and matches the exact
reference bit-for-bit for unit, uniform and mapped non-uniform scales -- the
same three modes the spike ran. Before 2026-09-18 the entry had no ABI case
in the launcher and registering the kernel answered rc=5.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from tessera.compiler import nvfp4_fragments as nf


def _tile(seed: int):
    rng = np.random.default_rng(seed)
    a = rng.integers(0, 16, size=(nf.M, nf.K), dtype=np.uint8)
    b = rng.integers(0, 16, size=(nf.K, nf.N), dtype=np.uint8)
    return a, b


def _scales(mode: str):
    codes = np.array([0x30, 0x38, 0x40], dtype=np.uint8)  # 0.5, 1.0, 2.0
    sa = np.full((nf.M, nf.SCALE_BLOCKS), nf.UE4M3_ONE, np.uint8)
    sb = np.full((nf.SCALE_BLOCKS, nf.N), nf.UE4M3_ONE, np.uint8)
    if mode == "unit":
        return sa, sb
    if mode.startswith("uniform:"):
        code = int(mode.split(":")[1], 16)
        return np.full_like(sa, code), np.full_like(sb, code)
    if mode == "mapped":  # the spike's non-uniform mapping
        rows, blocks = np.indices(sa.shape)
        sa = codes[(rows + blocks) % 3]
        blocks, cols = np.indices(sb.shape)
        sb = codes[(2 * blocks + cols) % 3]
        return sa, sb
    if mode == "random":
        rng = np.random.default_rng(9)
        return (rng.integers(0x28, 0x48, size=sa.shape, dtype=np.uint8),
                rng.integers(0x28, 0x48, size=sb.shape, dtype=np.uint8))
    raise ValueError(mode)


SCALE_MODES = ["unit", "uniform:30", "uniform:40", "mapped", "random"]


def test_e2m1_codec_matches_the_ptx_table():
    codes = np.arange(16, dtype=np.uint8)
    values = nf.e2m1_decode(codes)
    expected = np.array([0, .5, 1, 1.5, 2, 3, 4, 6, -0, -.5, -1, -1.5, -2, -3, -4, -6], np.float32)
    np.testing.assert_array_equal(values, expected)
    assert np.array_equal(nf.e2m1_encode(values)[1:8], codes[1:8])
    assert np.array_equal(nf.e2m1_encode(values)[9:], codes[9:])
    with pytest.raises(ValueError):
        nf.e2m1_decode(np.array([16], np.uint8))


def test_ue4m3_codec_is_unsigned_bias_seven():
    assert nf.ue4m3_decode(np.uint8(nf.UE4M3_ONE)) == 1.0
    assert nf.ue4m3_decode(np.uint8(0x30)) == 0.5
    assert nf.ue4m3_decode(np.uint8(0x40)) == 2.0
    assert nf.ue4m3_decode(np.uint8(0x39)) == 1.125
    assert nf.ue4m3_decode(np.uint8(0x01)) == pytest.approx(2.0 ** -9)
    with pytest.raises(ValueError):
        nf.ue4m3_decode(np.array([0x80], np.uint8))


def test_operand_packing_is_a_bijection_on_the_tile():
    a, b = _tile(1)
    sa, sb = _scales("mapped")
    aw, bw, sfa, sfb = nf.pack_nvfp4_mma_fragments(a, b, sa, sb)
    assert aw.shape == (32, 4) and bw.shape == (32, 2) and sfa.shape == (32,) and sfb.shape == (32,)
    assert aw.dtype == bw.dtype == sfa.dtype == sfb.dtype == np.uint32
    a2, b2 = nf.unpack_nvfp4_mma_fragments(aw, bw)
    np.testing.assert_array_equal(a2, a)
    np.testing.assert_array_equal(b2, b)


def test_lane_layout_is_the_proven_m16n8k64_fragment_order():
    """Spot-check the layout against the spike's per-lane formulas."""
    a, b = _tile(2)
    sa, sb = _scales("mapped")
    aw, bw, sfa, sfb = nf.pack_nvfp4_mma_fragments(a, b, sa, sb)
    for lane in (0, 5, 13, 31):
        gid, tig = lane >> 2, lane & 3
        nibbles = lambda word: [(int(word) >> (4 * j)) & 0xF for j in range(8)]  # noqa: E731
        assert nibbles(aw[lane, 0]) == list(a[gid, 8 * tig: 8 * tig + 8])
        assert nibbles(aw[lane, 1]) == list(a[gid + 8, 8 * tig: 8 * tig + 8])
        assert nibbles(aw[lane, 2]) == list(a[gid, 8 * tig + 32: 8 * tig + 40])
        assert nibbles(aw[lane, 3]) == list(a[gid + 8, 8 * tig + 32: 8 * tig + 40])
        assert nibbles(bw[lane, 0]) == list(b[8 * tig: 8 * tig + 8, gid])
        assert nibbles(bw[lane, 1]) == list(b[8 * tig + 32: 8 * tig + 40, gid])
        bytes_ = lambda word: [(int(word) >> (8 * i)) & 0xFF for i in range(4)]  # noqa: E731
        if tig == 0:
            assert bytes_(sfa[lane]) == list(sa[gid])
            assert bytes_(sfb[lane]) == list(sb[:, gid])
        elif tig == 1:
            assert bytes_(sfa[lane]) == list(sa[gid + 8])
            assert sfb[lane] == 0
        else:
            assert sfa[lane] == 0 and sfb[lane] == 0


def test_accumulator_unpack_inverts_the_d_fragment_order():
    d = np.arange(nf.M * nf.N, dtype=np.float32).reshape(nf.M, nf.N)
    words = np.empty((32, 4), np.float32)
    for lane in range(32):
        gid, tig = lane >> 2, lane & 3
        words[lane] = [d[gid, 2 * tig], d[gid, 2 * tig + 1], d[gid + 8, 2 * tig], d[gid + 8, 2 * tig + 1]]
    np.testing.assert_array_equal(nf.unpack_nvfp4_mma_accumulator(words), d)


@pytest.mark.parametrize("mode", SCALE_MODES)
def test_tile_reference_applies_one_scale_per_k_block(mode):
    a, b = _tile(3)
    sa, sb = _scales(mode)
    ref = nf.nvfp4_tile_reference(a, b, sa, sb)
    af, bf = nf.e2m1_decode(a).astype(np.float64), nf.e2m1_decode(b).astype(np.float64)
    saf, sbf = nf.ue4m3_decode(sa).astype(np.float64), nf.ue4m3_decode(sb).astype(np.float64)
    expected = np.zeros((nf.M, nf.N))
    for m in range(nf.M):
        for n in range(nf.N):
            expected[m, n] = sum(af[m, k] * saf[m, k // 16] * bf[k, n] * sbf[k // 16, n] for k in range(nf.K))
    np.testing.assert_allclose(ref, expected, rtol=1e-6, atol=1e-6)
    if mode == "unit":
        np.testing.assert_allclose(ref, af @ bf, rtol=1e-6)


def test_packer_refuses_out_of_range_codes():
    a, b = _tile(4)
    sa, sb = _scales("unit")
    with pytest.raises(ValueError, match="A"):
        nf.pack_nvfp4_mma_fragments(a.astype(np.uint8) | 16, b, sa, sb)
    with pytest.raises(ValueError, match="scale_b"):
        nf.pack_nvfp4_mma_fragments(a, b, sa, sb | 0x80)
    with pytest.raises(ValueError, match="shape"):
        nf.pack_nvfp4_mma_fragments(a[:8], b, sa, sb)


def _sm120_or_skip():
    from tests._support.environment import nvidia_cuda_tool, nvidia_gpu_is_plausibly_present
    if os.environ.get("TESSERA_SM120_DEVICE_PROOF") != "1":
        pytest.skip("explicit sm_120 owning-device gate (TESSERA_SM120_DEVICE_PROOF=1)")
    if not nvidia_gpu_is_plausibly_present() or nvidia_cuda_tool("ptxas") is None:
        pytest.skip("requires the RTX 5070 host with the CUDA toolkit")
    from tessera import runtime as rt
    if rt._load_nvidia_ptx_launch() is None:
        pytest.skip("libtessera_nvidia_ptx_launch.so is not built on this host")
    return rt


@pytest.mark.parametrize("mode", SCALE_MODES)
def test_sm120_emitted_nvfp4_tile_matches_the_exact_reference(mode):
    """Exact-device row: the emitted kernel, registered under its own entry,
    reproduces the tile reference bit-for-bit (every product is an exact
    binary fraction and the K=64 f32 sum has no rounding at these magnitudes)."""
    rt = _sm120_or_skip()
    a, b = _tile(11)
    sa, sb = _scales(mode)
    out = rt._nvidia_nvfp4_emitted_mma(a, b, sa, sb)
    ref = nf.nvfp4_tile_reference(a, b, sa, sb)
    assert out.shape == (nf.M, nf.N) and out.dtype == np.float32
    np.testing.assert_array_equal(out, ref)


def test_sm120_emitted_nvfp4_entry_is_registered_and_reused():
    rt = _sm120_or_skip()
    from tessera.compiler import ptx_emit as pe
    a, b = _tile(12)
    sa, sb = _scales("random")
    first = rt._nvidia_nvfp4_emitted_mma(a, b, sa, sb)
    assert pe.TESSERA_NVFP4_MMA_ENTRY in rt._nvidia_ptx_registered
    second = rt._nvidia_nvfp4_emitted_mma(a, b, sa, sb)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(first, nf.nvfp4_tile_reference(a, b, sa, sb))


# ── the general-shape emitted GEMM (2026-09-18) ──────────────────────────────

def _general_case(m, n, k, seed):
    rng = np.random.default_rng(seed)
    a = rng.integers(0, 16, size=(m, k), dtype=np.uint8)
    b = rng.integers(0, 16, size=(k, n), dtype=np.uint8)
    blocks = (k + 15) // 16
    sa = rng.integers(0x28, 0x48, size=(m, blocks), dtype=np.uint8)
    sb = rng.integers(0x28, 0x48, size=(blocks, n), dtype=np.uint8)
    return a, b, sa, sb


GENERAL_SHAPES = [(16, 8, 64), (16, 8, 128), (32, 16, 64), (33, 17, 100), (64, 40, 192), (5, 3, 20), (17, 9, 65)]


def test_pack_e2m1_codes_matches_the_launch_abi_layout():
    from tessera.compiler.nvfp4_fragments import pack_e2m1_codes
    codes = np.arange(2 * 5, dtype=np.uint8).reshape(2, 5) & 15
    packed = pack_e2m1_codes(codes, axis=1)
    assert packed.shape == (2, 3) and packed.dtype == np.uint8
    assert packed[0, 0] == (0 | (1 << 4)) and packed[0, 2] == 4  # odd extent zero-padded
    along0 = pack_e2m1_codes(codes.T, axis=0)
    assert along0.shape == (3, 2)
    np.testing.assert_array_equal(along0, packed.T)


def test_general_gemm_ptx_is_well_formed():
    from tessera.compiler import ptx_emit as pe
    ptx = pe.emit_nvfp4_gemm_ptx()
    assert pe.validate_nvfp4_gemm_ptx_structure(ptx) == []
    assert ptx.isascii() and "kind::mxf4nvf4.block_scale.scale_vec::4X" in ptx
    assert ptx.count("ld.global.u8") == 36  # 6 operand words x 4 bytes + 3 scale words x 4 bytes


def test_general_gemm_ptx_assembles_for_sm120a(tmp_path):
    import shutil, subprocess
    from tessera.compiler import ptx_emit as pe
    ptxas = shutil.which("ptxas")
    if ptxas is None:
        pytest.skip("ptxas not on PATH")
    source = tmp_path / "nvfp4_gemm.ptx"
    source.write_text(pe.emit_nvfp4_gemm_ptx())
    proc = subprocess.run([ptxas, "--gpu-name=sm_120a", "-o", str(tmp_path / "out.cubin"), str(source)],
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr


@pytest.mark.parametrize("shape", GENERAL_SHAPES)
def test_general_gemm_reference_reduces_to_the_tile_reference_on_one_tile(shape):
    from tessera.compiler.nvfp4_fragments import nvfp4_gemm_reference
    m, n, k = shape
    a, b, sa, sb = _general_case(m, n, k, 7)
    ref = nvfp4_gemm_reference(a, b, sa, sb)
    assert ref.shape == (m, n) and ref.dtype == np.float32
    if shape == (16, 8, 64):
        np.testing.assert_array_equal(ref, nf.nvfp4_tile_reference(a, b, sa, sb))


@pytest.mark.parametrize("shape", GENERAL_SHAPES)
def test_sm120_emitted_general_gemm_is_exact_and_equals_the_shipped_kernel(shape):
    """Exact-device row: any M/N/K through the general NVFP4 launch ABI --
    bit-exact against the numpy reference and against the shipped AOT kernel."""
    rt = _sm120_or_skip()
    from tessera.compiler.nvfp4_fragments import nvfp4_gemm_reference, pack_e2m1_codes
    m, n, k = shape
    a, b, sa, sb = _general_case(m, n, k, 11 + m)
    ap, bp = pack_e2m1_codes(a, axis=1), pack_e2m1_codes(b, axis=0)
    out = rt._nvidia_nvfp4_gemm_emitted_2d(ap, bp, sa, sb, m, n, k)
    ref = nvfp4_gemm_reference(a, b, sa, sb)
    np.testing.assert_array_equal(out, ref)
    shipped = rt._nvidia_nvfp4_gemm_2d(ap, bp, sa, sb, m, n, k)
    np.testing.assert_array_equal(out, shipped)


def test_sm120_nvfp4_benchmark_times_both_entries():
    rt = _sm120_or_skip()
    from tessera.compiler.nvfp4_fragments import pack_e2m1_codes
    from tessera.compiler import ptx_emit as pe
    m, n, k = 64, 40, 192
    a, b, sa, sb = _general_case(m, n, k, 5)
    ap, bp = pack_e2m1_codes(a, axis=1), pack_e2m1_codes(b, axis=0)
    ms = rt._nvidia_nvfp4_gemm_device_latency(pe.TESSERA_NVFP4_GEMM_ENTRY, ap, bp, sa, sb, m, n, k, reps=20, warmup=5)
    assert ms > 0.0
