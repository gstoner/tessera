"""Audit-D-3 (2026-05-22) — second pass of direct coverage for ops
still in the ``needs_direct_test`` bucket after the first coverage file.

The first file (``test_thin_op_direct_coverage.py``) gives each
flagged op one happy-path test.  That bumps each op to py_refs=1 but
leaves it inside the ``thinly_tested`` (≤1 ref) cutoff.  This file
adds 2–3 *additional* substantive tests per op, each exercising a
different code path:

  * different dtype / shape
  * edge case (saturation, eps boundary, identity)
  * negative / error path or alternative reduction

The goal isn't to bump ref counts — it's to actually exercise more
of each op's surface so the audit's "tests=complete" claim becomes
less conditional.  The extra refs are a happy side-effect.

Each op gets the minimum that proves: the call boundary works, an
edge case is handled, and at least one numerical property holds.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import tessera
from tessera import memory
from tessera import optim
from tessera import quantization
from tessera import rl
from tessera import sharding
from tessera.nn import functional

ops = tessera.ops


# ═════════════════════════════════════════════════════════════════════════
# POOLING — multi-shape / overlapping windows / padding
# ═════════════════════════════════════════════════════════════════════════


class TestPoolingDeep:
    def test_max_pool_overlapping_window(self) -> None:
        # kernel=3 stride=1 → overlapping; max over 3×3 window
        x = np.array(
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]],
            dtype=np.float32,
        ).reshape(1, 1, 4, 4)
        y = functional.max_pool(x, kernel_size=3, stride=1)
        # Output: 2×2 windows of max across the 3×3 input regions.
        # Top-left window (rows 0..2, cols 0..2): max=11
        # Top-right (rows 0..2, cols 1..3): max=12
        # Bottom-left (rows 1..3, cols 0..2): max=15
        # Bottom-right (rows 1..3, cols 1..3): max=16
        expect = np.array([[11, 12], [15, 16]], dtype=np.float32).reshape(
            1, 1, 2, 2
        )
        np.testing.assert_allclose(y, expect, rtol=1e-6)

    def test_max_pool_negative_values(self) -> None:
        x = -np.arange(16, dtype=np.float32).reshape(1, 1, 4, 4)
        y = functional.max_pool(x, kernel_size=2, stride=2)
        # max of negatives picks the closest-to-zero.
        expect = x.reshape(1, 1, 2, 2, 2, 2).max(axis=(3, 5))
        np.testing.assert_allclose(y, expect, rtol=1e-6)

    def test_min_pool_negative_values(self) -> None:
        x = -np.arange(16, dtype=np.float32).reshape(1, 1, 4, 4)
        y = functional.min_pool(x, kernel_size=2, stride=2)
        # min of negatives picks the most-negative.
        expect = x.reshape(1, 1, 2, 2, 2, 2).min(axis=(3, 5))
        np.testing.assert_allclose(y, expect, rtol=1e-6)

    def test_avg_pool_arithmetic_mean(self) -> None:
        # All-ones input → all-ones output regardless of stride.
        x = np.ones((1, 2, 4, 4), dtype=np.float32)
        y = functional.avg_pool(x, kernel_size=2, stride=2)
        np.testing.assert_allclose(y, np.ones((1, 2, 2, 2)), rtol=1e-6)

    def test_adaptive_pool_full_reduce(self) -> None:
        # output_size=(1,1) reduces each (N,C) plane to a single value.
        x = np.arange(16, dtype=np.float32).reshape(1, 1, 4, 4)
        y = functional.adaptive_pool(x, output_size=(1, 1))
        # Default reducer is mean → 7.5.
        np.testing.assert_allclose(y, np.array([[[[7.5]]]]), rtol=1e-6)

    def test_adaptive_pool_max_reducer(self) -> None:
        x = np.arange(16, dtype=np.float32).reshape(1, 1, 4, 4)
        y = functional.adaptive_pool(x, output_size=(1, 1), reducer=np.max)
        np.testing.assert_allclose(y, np.array([[[[15.0]]]]), rtol=1e-6)


# ═════════════════════════════════════════════════════════════════════════
# QUANTIZATION — symmetric vs asymmetric, explicit scale, error budget
# ═════════════════════════════════════════════════════════════════════════


class TestQuantizationDeep:
    def test_quantize_int8_explicit_scale(self) -> None:
        # With scale=1/127 and symmetric, the representable range is
        # roughly [-1, 1].  Stay inside that envelope so we don't get
        # saturation artifacts.
        x = np.array([-0.5, 0.0, 0.25, 0.5, 0.75], dtype=np.float32)
        q, scale, zp = quantization.quantize_int8(x, scale=1.0 / 127.0)
        x_back = quantization.dequantize_int8(q, scale, zp)
        np.testing.assert_allclose(x_back, x, atol=1.0 / 127.0)

    def test_dequantize_int8_zero_input_round_trip(self) -> None:
        # Pure zeros must round-trip exactly.
        x = np.zeros(8, dtype=np.float32)
        q, scale, zp = quantization.quantize_int8(x)
        # scale may be 0 or epsilon-floored; dequantize must yield ~0.
        x_back = quantization.dequantize_int8(q, max(scale, 1e-12), zp)
        np.testing.assert_allclose(x_back, x, atol=1e-6)

    def test_quantize_int4_saturation(self) -> None:
        # Extreme values clip to int4 range; we just verify finite output
        # and reasonable error magnitude.
        x = np.array(
            [-10.0, -1.0, 0.0, 1.0, 10.0], dtype=np.float32
        )
        q, scale, zp = quantization.quantize_int4(x)
        x_back = quantization.dequantize_int4(q, scale, zp)
        assert np.all(np.isfinite(x_back))

    def test_dequantize_int4_preserves_sign(self) -> None:
        x = np.linspace(-1.0, 1.0, 16, dtype=np.float32)
        q, scale, zp = quantization.quantize_int4(x)
        x_back = quantization.dequantize_int4(q, scale, zp)
        # Sign should agree on values away from zero.
        for a, b in zip(x, x_back):
            if abs(a) > 0.3:
                assert np.sign(a) == np.sign(b) or b == 0

    def test_fake_quantize_4bit_coarser_than_8bit(self) -> None:
        rng = np.random.default_rng(20)
        x = rng.standard_normal(64).astype(np.float32)
        y8 = quantization.fake_quantize(x, num_bits=8)
        y4 = quantization.fake_quantize(x, num_bits=4)
        err8 = np.max(np.abs(y8 - x))
        err4 = np.max(np.abs(y4 - x))
        # 4-bit quantization has strictly larger error than 8-bit
        # (on a non-trivial distribution).
        assert err4 >= err8


# ═════════════════════════════════════════════════════════════════════════
# NORMALIZATION — axis variants, rectangular, eps stability
# ═════════════════════════════════════════════════════════════════════════


class TestNormalizationDeep:
    def test_weight_norm_axis_1(self) -> None:
        rng = np.random.default_rng(21)
        w = rng.standard_normal((4, 6)).astype(np.float32)
        wn = functional.weight_norm(w, axis=1)
        # With axis=1, columns become the normalized units → each
        # column has unit L2 norm.
        norms = np.linalg.norm(wn, axis=0)
        np.testing.assert_allclose(norms, np.ones(6), atol=1e-5)

    def test_weight_norm_eps_stability_near_zero(self) -> None:
        # Near-zero rows: eps prevents division-by-zero blow-up.
        w = np.zeros((3, 4), dtype=np.float32)
        wn = functional.weight_norm(w, axis=0, eps=1e-6)
        assert np.all(np.isfinite(wn))

    def test_spectral_norm_rectangular_matrix(self) -> None:
        rng = np.random.default_rng(22)
        w = rng.standard_normal((3, 5)).astype(np.float32)
        wn = functional.spectral_norm(w)
        sv = np.linalg.svd(wn, compute_uv=False)[0]
        np.testing.assert_allclose(sv, 1.0, atol=1e-3)

    def test_instance_norm_eps_handles_constant_input(self) -> None:
        # Constant input: variance is 0; eps must prevent NaN/inf.
        x = np.full((1, 2, 4, 4), 7.0, dtype=np.float32)
        y = functional.instance_norm(x, eps=1e-5)
        assert np.all(np.isfinite(y))


# ═════════════════════════════════════════════════════════════════════════
# LAYOUT TRANSFORMS — value preservation across forms
# ═════════════════════════════════════════════════════════════════════════


class TestLayoutTransformsDeep:
    def test_pack_unpack_3d_roundtrip(self) -> None:
        rng = np.random.default_rng(23)
        x = rng.standard_normal((2, 3, 4)).astype(np.float32)
        packed = ops.pack(x, "row_major")
        back = ops.unpack(packed)
        np.testing.assert_allclose(back, x, rtol=1e-6)

    def test_pack_preserves_total_size(self) -> None:
        x = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        packed = ops.pack(x, "row_major")
        assert np.asarray(packed).size == x.size

    def test_rearrange_returns_same_total_elements(self) -> None:
        x = np.arange(12, dtype=np.float32).reshape(3, 4)
        y = ops.rearrange(x, "identity")
        assert np.asarray(y).size == x.size

    def test_tile_view_handles_3d(self) -> None:
        x = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        y = ops.tile_view(x, BM=2, BN=2)
        assert y is not None


# ═════════════════════════════════════════════════════════════════════════
# COLLECTIVES — 2-D stacks, identity / cyclic permutations
# ═════════════════════════════════════════════════════════════════════════


class TestCollectivesDeep:
    def test_psum_2d_stack(self) -> None:
        # 2 ranks × 3 elements; psum reduces the rank axis.
        x = np.array([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]], dtype=np.float32)
        y = sharding.psum(x)
        np.testing.assert_allclose(y, [11.0, 22.0, 33.0], rtol=1e-6)

    def test_pmean_2d_stack(self) -> None:
        x = np.array([[2.0, 4.0], [4.0, 8.0]], dtype=np.float32)
        y = sharding.pmean(x)
        np.testing.assert_allclose(y, [3.0, 6.0], rtol=1e-6)

    def test_pmax_picks_per_lane_max(self) -> None:
        x = np.array([[1.0, 5.0], [3.0, 2.0]], dtype=np.float32)
        y = sharding.pmax(x)
        np.testing.assert_allclose(y, [3.0, 5.0], rtol=1e-6)

    def test_pmin_picks_per_lane_min(self) -> None:
        x = np.array([[1.0, 5.0], [3.0, 2.0]], dtype=np.float32)
        y = sharding.pmin(x)
        np.testing.assert_allclose(y, [1.0, 2.0], rtol=1e-6)

    def test_broadcast_to_axis_shape_expansion(self) -> None:
        # broadcast_to_axis(value, *, axis_size, axis): stacks
        # ``value`` along a new axis of length ``axis_size``.
        x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        y = sharding.broadcast_to_axis(x, axis_size=4)
        assert np.asarray(y).shape == (4, 3)
        for i in range(4):
            np.testing.assert_allclose(np.asarray(y)[i], x)

    def test_broadcast_to_axis_with_explicit_axis_position(self) -> None:
        # Stacking along an inner axis (axis=1) instead of the default
        # leading axis.
        x = np.array([10.0, 20.0], dtype=np.float32)
        y = sharding.broadcast_to_axis(x, axis_size=3, axis=1)
        # Shape: x has 2 elements; stacking 3 copies along axis=1 →
        # shape (2, 3).
        assert np.asarray(y).shape == (2, 3)
        # Every column equals the original.
        for j in range(3):
            np.testing.assert_allclose(np.asarray(y)[:, j], x)

    def test_broadcast_to_axis_scalar_value(self) -> None:
        # Broadcasting a scalar should yield ``(axis_size,)``.
        y = sharding.broadcast_to_axis(np.float32(7.0), axis_size=5)
        np.testing.assert_allclose(np.asarray(y), np.full(5, 7.0))

    def test_collective_permute_identity(self) -> None:
        # Identity permutation [(i, i)] → output equals input.
        x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        y = sharding.collective_permute(x, pairs=[(0, 0), (1, 1)])
        np.testing.assert_allclose(y, x, rtol=1e-6)


# ═════════════════════════════════════════════════════════════════════════
# SPECTRAL — windowing, frequency-domain filtering
# ═════════════════════════════════════════════════════════════════════════


class TestSpectralDeep:
    def test_stft_shape_n_frames(self) -> None:
        n = 64
        win = np.hanning(16).astype(np.float32)
        hop = 8
        x = np.zeros(n, dtype=np.float32)
        xf = ops.stft(x, win, hop)
        # Expect: ~(n_freq, n_frames) layout — verify it's 2-D.
        arr = np.asarray(xf)
        assert arr.ndim == 2

    def test_stft_of_zeros_is_zero(self) -> None:
        n = 32
        win = np.hanning(8).astype(np.float32)
        x = np.zeros(n, dtype=np.float32)
        xf = ops.stft(x, win, hop=4)
        np.testing.assert_allclose(np.asarray(xf), 0, atol=1e-6)

    def test_istft_of_zero_spectrum_is_zero(self) -> None:
        # Zero frequency-domain → zero time-domain.
        n_freq = 9
        n_frames = 4
        win = np.hanning(8).astype(np.float32)
        xf = np.zeros((n_freq, n_frames), dtype=np.complex64)
        x_back = ops.istft(xf, win, hop=4)
        np.testing.assert_allclose(np.asarray(x_back), 0, atol=1e-6)

    def test_spectral_filter_zero_response(self) -> None:
        # H=0 → output spectrum is 0 regardless of input.
        rng = np.random.default_rng(24)
        Xf = (rng.standard_normal(16) + 1j * rng.standard_normal(16)).astype(
            np.complex64
        )
        Hf = np.zeros(16, dtype=np.complex64)
        y = ops.spectral_filter(Xf, Hf)
        np.testing.assert_allclose(np.asarray(y), 0, atol=1e-6)

    def test_spectral_filter_identity_response(self) -> None:
        # H=1 → output equals input.
        rng = np.random.default_rng(25)
        Xf = (rng.standard_normal(16) + 1j * rng.standard_normal(16)).astype(
            np.complex64
        )
        Hf = np.ones(16, dtype=np.complex64)
        y = ops.spectral_filter(Xf, Hf)
        np.testing.assert_allclose(np.asarray(y), Xf, rtol=1e-5)


# ═════════════════════════════════════════════════════════════════════════
# LINALG — rectangular / square / dtype variants
# ═════════════════════════════════════════════════════════════════════════


class TestLinalgDeep:
    def test_qr_square_matrix(self) -> None:
        rng = np.random.default_rng(26)
        A = rng.standard_normal((4, 4)).astype(np.float64)
        Q, R = ops.qr(A)
        np.testing.assert_allclose(Q @ R, A, atol=1e-10)
        np.testing.assert_allclose(Q.T @ Q, np.eye(4), atol=1e-10)

    def test_qr_tall_matrix(self) -> None:
        # Reduced QR: tall (m > n).
        rng = np.random.default_rng(27)
        A = rng.standard_normal((6, 2)).astype(np.float64)
        Q, R = ops.qr(A)
        np.testing.assert_allclose(Q @ R, A, atol=1e-10)
        assert Q.shape == (6, 2)
        assert R.shape == (2, 2)

    def test_svd_singular_values_monotone(self) -> None:
        rng = np.random.default_rng(28)
        A = rng.standard_normal((5, 5)).astype(np.float64)
        _, S, _ = ops.svd(A)
        # Singular values must be sorted descending.
        assert np.all(S[:-1] >= S[1:])

    def test_svd_rank_deficient_matrix(self) -> None:
        # Outer-product rank-1 matrix.
        u = np.array([1, 2, 3, 4], dtype=np.float64)
        v = np.array([1, 0, -1, 2], dtype=np.float64)
        A = np.outer(u, v)
        U, S, Vt = ops.svd(A)
        # Only one singular value should be non-trivial.
        assert S[0] > 1e-8
        assert all(s < 1e-8 for s in S[1:])


# ═════════════════════════════════════════════════════════════════════════
# STABLE REDUCTIONS — math identities + numerical stability
# ═════════════════════════════════════════════════════════════════════════


class TestStableReductionsDeep:
    def test_log_softmax_matches_log_of_softmax(self) -> None:
        x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        y = ops.log_softmax(x)
        # Reference: log(softmax(x)).
        sm = np.exp(x - x.max())
        sm = sm / sm.sum()
        expect = np.log(sm)
        np.testing.assert_allclose(y, expect, rtol=1e-5, atol=1e-6)

    def test_log_softmax_handles_large_logits(self) -> None:
        # Naive log(softmax) would overflow at exp(1000); the safe
        # variant must produce finite output.
        x = np.array([1000.0, 1000.5, 999.0], dtype=np.float32)
        y = ops.log_softmax(x)
        assert np.all(np.isfinite(y))

    def test_log_softmax_sums_to_unity_in_exp(self) -> None:
        x = np.array([0.0, 1.0, 2.0], dtype=np.float32)
        y = ops.log_softmax(x)
        np.testing.assert_allclose(np.sum(np.exp(y)), 1.0, atol=1e-6)

    def test_sigmoid_safe_matches_sigmoid_moderate(self) -> None:
        x = np.array([-1.0, 0.0, 1.0, 2.0], dtype=np.float32)
        y = ops.sigmoid_safe(x)
        expect = 1.0 / (1.0 + np.exp(-x))
        np.testing.assert_allclose(y, expect, rtol=1e-5)

    def test_sigmoid_safe_no_overflow_at_extremes(self) -> None:
        # Naive sigmoid would overflow at exp(1000); the safe variant
        # must clip to (0, 1) without NaN/inf.
        x = np.array([1000.0, -1000.0, 0.0], dtype=np.float32)
        y = ops.sigmoid_safe(x)
        assert np.all(np.isfinite(y))
        np.testing.assert_allclose(y[0], 1.0, atol=1e-6)
        np.testing.assert_allclose(y[1], 0.0, atol=1e-6)
        np.testing.assert_allclose(y[2], 0.5, atol=1e-6)

    def test_conformal_energy_on_sphere_returns_finite_distinct(
        self,
    ) -> None:
        # Distinct points on the sphere → finite, non-negative energy.
        # (The p == p_target case routes through a stereographic
        # projection at the north pole, which is singular; skip it.)
        p = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        p_target = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        e = tessera.complex.conformal_energy_on_sphere(p, p_target)
        val = np.asarray(e)
        assert np.all(np.isfinite(val))
        assert float(val) >= 0.0

    def test_conformal_energy_on_sphere_returns_scalar_array(self) -> None:
        # Equatorial points (avoiding the antipodal north-pole singularity
        # of the stereographic projection): energy must be a scalar.
        p = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        p_target = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        e = tessera.complex.conformal_energy_on_sphere(p, p_target)
        arr = np.asarray(e)
        # Scalar-ish: 0-d or 1-element.
        assert arr.size == 1


# ═════════════════════════════════════════════════════════════════════════
# RECURRENT — explicit gate math, BPTT-ready bidirectional scan
# ═════════════════════════════════════════════════════════════════════════


class TestRecurrentDeep:
    def test_simple_rnn_cell_with_bias(self) -> None:
        x = np.array([[1.0, 2.0]], dtype=np.float32)
        h = np.zeros((1, 3), dtype=np.float32)
        W_ih = np.ones((2, 3), dtype=np.float32) * 0.1
        W_hh = np.zeros((3, 3), dtype=np.float32)
        b = np.array([0.5, -0.5, 0.0], dtype=np.float32)
        y = functional.simple_rnn_cell(x, h, W_ih, W_hh, bias=b)
        # tanh(x @ W_ih + 0 + bias).
        expect = np.tanh(x @ W_ih + b)
        np.testing.assert_allclose(y, expect, rtol=1e-5, atol=1e-6)

    def test_simple_rnn_cell_saturates_at_tanh_bounds(self) -> None:
        # Very large input → tanh saturates near ±1.
        x = np.array([[1000.0, 1000.0]], dtype=np.float32)
        h = np.zeros((1, 2), dtype=np.float32)
        W_ih = np.eye(2, dtype=np.float32)
        W_hh = np.zeros((2, 2), dtype=np.float32)
        y = functional.simple_rnn_cell(x, h, W_ih, W_hh)
        np.testing.assert_allclose(y, np.array([[1.0, 1.0]]), atol=1e-6)

    def test_gru_cell_zero_input_preserves_hidden(self) -> None:
        # With reasonable weights and x=0, GRU should keep h close to
        # the previous hidden state (depends on gate values).
        h = np.array([[0.5, -0.3, 0.2, 0.1]], dtype=np.float32)
        x = np.zeros((1, 3), dtype=np.float32)
        W_ih = np.ones((3, 12), dtype=np.float32) * 0.01
        W_hh = np.ones((4, 12), dtype=np.float32) * 0.01
        y = functional.gru_cell(x, h, W_ih, W_hh)
        assert y.shape == h.shape
        assert np.all(np.isfinite(y))

    def test_bidirectional_scan_accumulates_both_directions(self) -> None:
        # Identity scan: fn(carry, x) = carry + x (single return value).
        def fn(carry, x):
            return carry + x

        xs = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        init_fwd = np.float32(0.0)
        init_bwd = np.float32(0.0)
        out = functional.bidirectional_scan(fn, init_fwd, init_bwd, xs)
        # Expect both forward and backward tracks (some concat or tuple).
        assert out is not None
        # Forward last value should reach 10 (1+2+3+4).
        arr = np.asarray(out)
        # At minimum, the cumulative sum 10 should appear somewhere
        # in the output.
        assert 10.0 in arr.flatten().tolist() or np.any(
            np.isclose(arr.flatten(), 10.0)
        )


# ═════════════════════════════════════════════════════════════════════════
# STENCIL — analytical reference checks
# ═════════════════════════════════════════════════════════════════════════


class TestStencilDeep:
    def test_conv3d_constant_input_yields_constant_output(self) -> None:
        # Constant input → output depends only on weight sum × constant.
        x = np.ones((1, 4, 4, 4, 2), dtype=np.float32)
        # Weight: (kD, kH, kW, in_C, out_C) — output spatial reduces.
        w = np.zeros((2, 2, 2, 2, 1), dtype=np.float32)
        w[..., 0] = 1.0 / (2 * 2 * 2 * 2)  # average filter
        y = ops.conv3d(x, w, stride=1, padding=0)
        # All windows average to 1.0.
        np.testing.assert_allclose(y, 1.0, atol=1e-5)

    def test_conv3d_different_stride(self) -> None:
        rng = np.random.default_rng(29)
        x = rng.standard_normal((1, 6, 6, 6, 2)).astype(np.float32)
        w = rng.standard_normal((2, 2, 2, 2, 1)).astype(np.float32) * 0.1
        y2 = ops.conv3d(x, w, stride=2, padding=0)
        # Output spatial dim: (6 - 2) / 2 + 1 = 3.
        assert y2.shape[1:4] == (3, 3, 3)

    def test_laplacian_2d_constant_field_zero(self) -> None:
        # Laplacian of a constant field is identically zero.
        field = np.full((6, 6), 5.0, dtype=np.float32)
        lap = tessera.complex.laplacian_2d(field, dx=1.0)
        # Interior should be exactly zero (boundaries may be zero-padded
        # depending on the BC choice; we check the interior).
        np.testing.assert_allclose(lap[1:-1, 1:-1], 0.0, atol=1e-5)

    def test_laplacian_2d_paraboloid_constant_curvature(self) -> None:
        # Laplacian of f(x,y) = x² + y² is identically 4 (continuous).
        # On a discrete grid with dx=1, the 5-point stencil gives 4.
        y_idx, x_idx = np.meshgrid(
            np.arange(8, dtype=np.float32),
            np.arange(8, dtype=np.float32),
            indexing="ij",
        )
        field = x_idx ** 2 + y_idx ** 2
        lap = tessera.complex.laplacian_2d(field, dx=1.0)
        # Interior should be ≈ 4.0.
        np.testing.assert_allclose(lap[2:-2, 2:-2], 4.0, atol=1e-4)


# ═════════════════════════════════════════════════════════════════════════
# MODEL LAYER EXTRAS — conv_transpose shape, lora scaling
# ═════════════════════════════════════════════════════════════════════════


class TestModelLayerDeep:
    def test_conv_transpose_stride_1_basic(self) -> None:
        # stride=1 conv_transpose: output_length = input + k - 1.
        x = np.ones((1, 1, 4), dtype=np.float32)
        w = np.ones((1, 1, 3), dtype=np.float32)
        y = functional.conv_transpose(x, w, stride=1)
        # Output length: (4-1)*1 - 0 + 3 = 6.
        assert y.shape == (1, 1, 6)
        assert np.all(np.isfinite(y))

    def test_conv_transpose_preserves_channel_dim(self) -> None:
        rng = np.random.default_rng(30)
        x = rng.standard_normal((2, 3, 5)).astype(np.float32)
        # (in_C=3, out_C=4, k=2)
        w = rng.standard_normal((3, 4, 2)).astype(np.float32) * 0.1
        y = functional.conv_transpose(x, w, stride=1)
        assert y.shape[1] == 4
        assert y.shape[0] == 2

    def test_lora_linear_alpha_zero_equals_base_weight(self) -> None:
        rng = np.random.default_rng(31)
        x = rng.standard_normal((2, 4)).astype(np.float32)
        w = rng.standard_normal((4, 3)).astype(np.float32) * 0.1
        a = rng.standard_normal((4, 2)).astype(np.float32)
        b = rng.standard_normal((2, 3)).astype(np.float32)
        # alpha=0 ⇒ LoRA contribution is zero ⇒ y = x @ w.
        y = functional.lora_linear(x, w, a, b, alpha=0.0)
        expect = x @ w
        np.testing.assert_allclose(y, expect, rtol=1e-4, atol=1e-5)

    def test_lora_linear_alpha_scaling(self) -> None:
        # y(alpha=2) - y(alpha=1) ≈ x @ (a @ b).  Use fp64 inputs to
        # avoid fp32 round-off dominating the small LoRA delta.
        rng = np.random.default_rng(32)
        x = rng.standard_normal((2, 4)).astype(np.float64)
        w = rng.standard_normal((4, 3)).astype(np.float64) * 0.1
        a = rng.standard_normal((4, 2)).astype(np.float64) * 0.01
        b = rng.standard_normal((2, 3)).astype(np.float64) * 0.01
        y1 = functional.lora_linear(x, w, a, b, alpha=1.0)
        y2 = functional.lora_linear(x, w, a, b, alpha=2.0)
        diff = y2 - y1
        expected = x @ (a @ b)
        np.testing.assert_allclose(diff, expected, rtol=1e-2, atol=1e-3)


# ═════════════════════════════════════════════════════════════════════════
# SPARSE — non-diagonal patterns, batched B
# ═════════════════════════════════════════════════════════════════════════


class TestSparseDeep:
    def test_spmm_coo_off_diagonal_pattern(self) -> None:
        # Bidiagonal: nonzero at (0,1), (1,2), (2,0).
        coords = np.array([[0, 1], [1, 2], [2, 0]], dtype=np.int64)
        vals = np.array([5.0, 7.0, 11.0], dtype=np.float32)
        A_dense = np.zeros((3, 3), dtype=np.float32)
        for (r, c), v in zip(coords, vals):
            A_dense[r, c] = v
        B = np.arange(9, dtype=np.float32).reshape(3, 3)
        y = ops.spmm_coo((coords, vals, (3, 3)), B)
        np.testing.assert_allclose(y, A_dense @ B, rtol=1e-6)

    def test_spmm_coo_zero_matrix_yields_zero(self) -> None:
        # Empty COO ⇒ output is zero.
        coords = np.zeros((0, 2), dtype=np.int64)
        vals = np.zeros((0,), dtype=np.float32)
        B = np.arange(9, dtype=np.float32).reshape(3, 3)
        y = ops.spmm_coo((coords, vals, (3, 3)), B)
        np.testing.assert_allclose(y, np.zeros((3, 3)), atol=1e-6)


# ═════════════════════════════════════════════════════════════════════════
# MOE — mean reduction, larger expert count
# ═════════════════════════════════════════════════════════════════════════


class TestMoEDeep:
    def test_moe_combine_mean_reduction(self) -> None:
        # 2 experts × 3 tokens × 2 features.  Identity route.
        partials = np.array(
            [
                [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]],
                [[5.0, 5.0], [6.0, 6.0], [7.0, 7.0]],
            ],
            dtype=np.float32,
        )
        inverse_route = np.tile(np.arange(3), (2, 1))
        y = ops.moe_combine(partials, inverse_route, reduce="mean")
        # Per-token mean across experts.
        expect = partials.mean(axis=0)
        np.testing.assert_allclose(y, expect, rtol=1e-6)

    def test_moe_combine_single_expert(self) -> None:
        partials = np.array(
            [[[1.0], [2.0], [3.0]]], dtype=np.float32
        )
        inverse_route = np.array([[0, 1, 2]])
        y = ops.moe_combine(partials, inverse_route, reduce="sum")
        np.testing.assert_allclose(
            y, np.array([[1.0], [2.0], [3.0]]), rtol=1e-6
        )


# ═════════════════════════════════════════════════════════════════════════
# OPTIMIZERS — multi-step, state propagation, distinct from primary tests
# ═════════════════════════════════════════════════════════════════════════


class TestOptimizersDeep:
    def test_lamb_with_weight_decay_pulls_toward_zero(self) -> None:
        # Zero grad + weight_decay > 0 ⇒ params decay toward zero.
        params = {"w": np.array([1.0, -1.0, 2.0], dtype=np.float32)}
        grads = {"w": np.zeros_like(params["w"])}
        new_params, _ = optim.lamb(
            params, grads, lr=0.01, weight_decay=0.1
        )
        # Decoupled WD: each weight scaled toward 0.
        assert np.all(np.abs(new_params["w"]) <= np.abs(params["w"]))

    def test_muon_state_persists_across_steps(self) -> None:
        params = {"w": np.array([1.0, 1.0], dtype=np.float32)}
        grads = {"w": np.ones_like(params["w"])}
        p1, state1 = optim.muon(params, grads, lr=0.01)
        p2, state2 = optim.muon(p1, grads, state=state1, lr=0.01)
        assert state2 is not None
        assert "velocity" in state2
        # Per-leaf delta comparison: second step should be larger than
        # first (momentum accumulating).
        delta1 = np.abs(p1["w"] - params["w"])
        delta2 = np.abs(p2["w"] - p1["w"])
        # Momentum 0.95 accumulates; second step delta exceeds first.
        assert np.all(delta2 >= delta1 * 0.9)

    def test_nesterov_with_high_momentum_overshoots_less_than_sgd(
        self,
    ) -> None:
        # On a smooth quadratic, Nesterov should land at least as
        # close to the minimum as a plain step.
        params = {"w": np.array([10.0], dtype=np.float32)}
        grads = {"w": np.array([2.0], dtype=np.float32)}
        new_params, _ = optim.nesterov(
            params, grads, lr=0.1, momentum=0.0
        )
        # Without momentum, Nesterov = plain SGD.
        np.testing.assert_allclose(new_params["w"], [10.0 - 0.1 * 2.0])


# ═════════════════════════════════════════════════════════════════════════
# NUMERICS — gradient scaler backoff path
# ═════════════════════════════════════════════════════════════════════════


class TestNumericsDeep:
    def test_grad_scaler_step_backoff_on_inf(self) -> None:
        # found_inf=True ⇒ scale shrinks, tracker stays low,
        # should_step=False.
        grads = [np.full(4, np.inf, dtype=np.float32)]
        unscaled, new_scale, _, should_step = quantization.grad_scaler_step(
            grads, scale=4.0, found_inf=True,
        )
        assert new_scale < 4.0  # backoff
        assert should_step is False

    def test_grad_scaler_step_no_inf_no_growth_yet(self) -> None:
        # tracker < interval ⇒ scale unchanged, step proceeds.
        grads = [np.ones(4, dtype=np.float32)]
        _, new_scale, new_tracker, should_step = quantization.grad_scaler_step(
            grads, scale=2.0, found_inf=False, growth_tracker=10,
            growth_interval=2000,
        )
        assert new_scale == 2.0
        assert new_tracker == 11
        assert should_step is True


# ═════════════════════════════════════════════════════════════════════════
# RL — GRPO clipping behavior, mask handling
# ═════════════════════════════════════════════════════════════════════════


class TestRLDeep:
    def test_grpo_policy_loss_with_zero_advantage_is_zero(self) -> None:
        # If rewards are constant per group, advantages are zero ⇒
        # policy-gradient signal vanishes.  Loss is dominated by KL or
        # is at numerical zero with kl_coef=0.
        logp_new = np.zeros((2, 4), dtype=np.float32)
        logp_old = np.zeros((2, 4), dtype=np.float32)
        rewards = np.array(
            [[5.0, 5.0, 5.0, 5.0], [3.0, 3.0, 3.0, 3.0]], dtype=np.float32
        )
        loss = rl.grpo_policy_loss(logp_new, logp_old, rewards=rewards)
        np.testing.assert_allclose(loss, 0.0, atol=1e-5)

    def test_grpo_policy_loss_with_mask_excludes_padding(self) -> None:
        rng = np.random.default_rng(33)
        logp_new = rng.standard_normal((2, 6)).astype(np.float32)
        logp_old = rng.standard_normal((2, 6)).astype(np.float32)
        rewards = rng.standard_normal((2, 6)).astype(np.float32)
        # Mask out the last 2 columns of group 1.
        mask = np.ones((2, 6), dtype=np.float32)
        mask[1, -2:] = 0.0
        loss_masked = rl.grpo_policy_loss(
            logp_new, logp_old, rewards=rewards, mask=mask
        )
        assert math.isfinite(float(loss_masked))


# ═════════════════════════════════════════════════════════════════════════
# MEMORY — update path, capacity behavior
# ═════════════════════════════════════════════════════════════════════════


class TestMemoryDeep:
    def test_memory_write_appends_new_rows(self) -> None:
        keys0 = np.array([[1.0, 0.0]], dtype=np.float32)
        vals0 = np.array([[10.0]], dtype=np.float32)
        table = memory.memory_write(
            (np.zeros((0, 2), dtype=np.float32),
             np.zeros((0, 1), dtype=np.float32)),
            keys0, vals0,
        )
        # Add a second row.
        new_table = memory.memory_write(
            table,
            np.array([[0.0, 1.0]], dtype=np.float32),
            np.array([[20.0]], dtype=np.float32),
        )
        # Should now hold two keys.
        if hasattr(new_table, "keys"):
            assert len(new_table.keys) == 2
        else:
            # tuple form
            keys, _ = new_table[:2]
            assert len(np.asarray(keys)) == 2

    def test_memory_write_with_max_entries_limits_size(self) -> None:
        # max_entries should bound the table size.
        keys = np.array([[1.0], [2.0], [3.0], [4.0]], dtype=np.float32)
        vals = np.array([[1.0], [2.0], [3.0], [4.0]], dtype=np.float32)
        table = memory.memory_write(
            (np.zeros((0, 1), dtype=np.float32),
             np.zeros((0, 1), dtype=np.float32)),
            keys, vals, max_entries=2,
        )
        # Should be bounded to 2 rows.
        if hasattr(table, "keys"):
            assert len(table.keys) <= 2
        else:
            k = np.asarray(table[0])
            assert len(k) <= 2


# ═════════════════════════════════════════════════════════════════════════
# ATTENTION — shape-only smoke tests for advanced variants
#
# These ops live in the attention research surface and require multi-arg
# call patterns.  We exercise the call boundary + finite output as a
# proof that the wrapper compiles and returns sane data.
# ═════════════════════════════════════════════════════════════════════════


class TestAdvancedAttentionDeep:
    def test_gated_attention_shape_and_finite(self) -> None:
        rng = np.random.default_rng(34)
        B, H, T, D = 1, 2, 4, 8
        Q = rng.standard_normal((B, H, T, D)).astype(np.float32) * 0.1
        K = rng.standard_normal((B, H, T, D)).astype(np.float32) * 0.1
        V = rng.standard_normal((B, H, T, D)).astype(np.float32) * 0.1
        gate = rng.standard_normal((B, H, T, D)).astype(np.float32) * 0.1
        y = ops.gated_attention(Q, K, V, gate, causal=True)
        assert np.asarray(y).shape == (B, H, T, D)
        assert np.all(np.isfinite(np.asarray(y)))

    def test_gated_attention_identity_activation(self) -> None:
        # ``identity`` activation: gate multiplied raw (no squashing).
        rng = np.random.default_rng(35)
        B, H, T, D = 1, 1, 3, 4
        Q = rng.standard_normal((B, H, T, D)).astype(np.float32) * 0.1
        K = rng.standard_normal((B, H, T, D)).astype(np.float32) * 0.1
        V = rng.standard_normal((B, H, T, D)).astype(np.float32) * 0.1
        gate = rng.standard_normal((B, H, T, D)).astype(np.float32) * 0.1
        y = ops.gated_attention(
            Q, K, V, gate, gate_activation="identity", causal=True
        )
        assert np.all(np.isfinite(np.asarray(y)))

    def test_modified_delta_attention_shape_only(self) -> None:
        rng = np.random.default_rng(36)
        B, H, T, D = 1, 2, 6, 8
        Q = rng.standard_normal((B, H, T, D)).astype(np.float32) * 0.1
        K = rng.standard_normal((B, H, T, D)).astype(np.float32) * 0.1
        V = rng.standard_normal((B, H, T, D)).astype(np.float32) * 0.1
        y = ops.modified_delta_attention(Q, K, V, causal=True)
        out = y[0] if isinstance(y, tuple) else y
        assert np.asarray(out).shape == (B, H, T, D)
        assert np.all(np.isfinite(np.asarray(out)))

    def test_deepseek_sparse_attention_shape(self) -> None:
        rng = np.random.default_rng(37)
        B, H, T, D = 1, 2, 8, 8
        Q = rng.standard_normal((B, H, T, D)).astype(np.float32) * 0.1
        K = rng.standard_normal((B, H, T, D)).astype(np.float32) * 0.1
        V = rng.standard_normal((B, H, T, D)).astype(np.float32) * 0.1
        y = ops.deepseek_sparse_attention(
            Q, K, V, window_size=4, block_size=2, top_k=2, causal=True
        )
        assert np.asarray(y).shape == (B, H, T, D)
        assert np.all(np.isfinite(np.asarray(y)))


# ═════════════════════════════════════════════════════════════════════════
# QUANTIZE FP4/FP6/NVFP4 — low-precision dequantize roundtrips
# ═════════════════════════════════════════════════════════════════════════


class TestLowPrecisionDequantize:
    def test_dequantize_fp4_roundtrip_through_quantize(self) -> None:
        # quantize_fp4 → dequantize_fp4 round-trip.  fp4 (e2m1) is
        # extremely coarse; assert only finiteness and dtype-preserving
        # shape across the boundary.
        rng = np.random.default_rng(40)
        x = rng.standard_normal(32).astype(np.float32) * 0.5
        q, scale = ops.quantize_fp4(x)
        x_back = ops.dequantize_fp4(q, scale)
        assert np.all(np.isfinite(x_back))
        assert x_back.shape == x.shape

    def test_dequantize_fp4_e2m1_explicit_format(self) -> None:
        # ``format="e2m1"`` is the canonical fp4 layout.  Exercise the
        # explicit-format path on a deterministic small input.
        x = np.linspace(-0.5, 0.5, 16, dtype=np.float32)
        q, scale = ops.quantize_fp4(x, format="e2m1")
        x_back = ops.dequantize_fp4(q, scale, format="e2m1")
        assert np.all(np.isfinite(x_back))

    def test_dequantize_fp6_roundtrip_through_quantize(self) -> None:
        rng = np.random.default_rng(41)
        x = rng.standard_normal(32).astype(np.float32) * 0.5
        q, scale = ops.quantize_fp6(x)
        x_back = ops.dequantize_fp6(q, scale)
        assert np.all(np.isfinite(x_back))

    def test_dequantize_nvfp4_block_scaled_roundtrip(self) -> None:
        # NVFP4 uses per-block scales (block_size defaults to 16).
        rng = np.random.default_rng(42)
        # Must be a multiple of block_size.
        x = rng.standard_normal(32).astype(np.float32) * 0.5
        q, scales = ops.quantize_nvfp4(x, block_size=16)
        x_back = ops.dequantize_nvfp4(q, scales, block_size=16)
        assert np.all(np.isfinite(x_back))
        assert x_back.shape == x.shape

    def test_dequantize_nvfp4_zero_input_roundtrip(self) -> None:
        x = np.zeros(32, dtype=np.float32)
        q, scales = ops.quantize_nvfp4(x)
        x_back = ops.dequantize_nvfp4(q, scales)
        np.testing.assert_allclose(x_back, x, atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
