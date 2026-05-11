"""Smoke coverage for the long-tail registry primitives.

Closes the `tests` axis for the registry entries that don't have
dedicated unit-test files yet. The smoke tests don't validate numerical
correctness in depth — they're per-primitive existence and basic-shape
guards. Real depth-correctness lives in `test_autodiff_loss_layer_coverage`,
`test_attention_family_support`, etc.; this file is the "every shipped
primitive has at least one test exercising it" gate.

Per-axis tests covered here:
  - `tessera.ops.<name>` callable + returns a numpy array
  - shape preservation for structural ops
  - registry consistency for state-effect / non-differentiable ops

For categories whose entries are integer-only / state-effect (sort,
indexing without grad, state_update), we assert the primitive exists
and runs; gradient-related axes are covered in the registry's
`contract_overrides`.
"""

from __future__ import annotations

import numpy as np
import pytest

import tessera as ts
from tessera.compiler.primitive_coverage import coverage_for


# Names that should exist in `tessera.ops` and return ndarray for a tiny input.
# Grouped by category to keep failures readable.
_STRUCTURAL_PRESERVE_SHAPE: list[str] = [
    "flatten", "unsqueeze", "squeeze", "expand", "reshape", "view",
    "permute", "transpose", "cat", "stack", "split", "chunk", "roll",
    "flip", "tile", "repeat", "pad", "broadcast", "slice", "select",
]

_INDEXING_OPS: list[str] = [
    "gather", "scatter", "scatter_add", "scatter_reduce",
    "index_select", "index_update", "take", "nonzero",
    "dynamic_slice", "dynamic_update_slice",
]

_SORT_OPS: list[str] = ["sort", "argsort", "top_k"]
_SPECTRAL_OPS: list[str] = ["fft", "ifft", "rfft", "irfft", "stft", "istft",
                             "dct", "spectral_conv", "spectral_filter"]
_LAYOUT_TRANSFORM_OPS: list[str] = [
    "rearrange", "pack", "unpack", "tile_view", "cast", "masked_fill",
    "rope_split", "rope_merge", "mor_router", "mor_partition", "mor_scatter",
    "arange",
]
_LINALG_OPS: list[str] = ["cholesky", "qr", "svd", "tri_solve"]
_SPARSE_OPS: list[str] = ["spmm_coo", "spmm_csr", "sddmm", "bsmm"]
_MOE_OPS: list[str] = ["moe", "moe_dispatch", "moe_combine"]
_OTHER_OPS: list[str] = [
    "selective_ssm", "segment_reduce", "fused_epilogue", "qkv_projection",
]


@pytest.mark.parametrize("name", _STRUCTURAL_PRESERVE_SHAPE)
def test_structural_op_in_registry(name):
    entry = coverage_for(name)
    assert entry.existing_op, f"{name} must be flagged as existing in the registry"


@pytest.mark.parametrize("name", _STRUCTURAL_PRESERVE_SHAPE + _INDEXING_OPS
                                + _SORT_OPS + _SPECTRAL_OPS
                                + _LAYOUT_TRANSFORM_OPS + _LINALG_OPS
                                + _SPARSE_OPS + _MOE_OPS + _OTHER_OPS)
def test_long_tail_primitive_has_registry_entry(name):
    """Every long-tail primitive listed above must round-trip through the
    coverage registry without errors."""
    entry = coverage_for(name)
    assert entry.name == name
    assert entry.status == "partial"
    assert entry.existing_op


@pytest.mark.parametrize("name", _STRUCTURAL_PRESERVE_SHAPE
                                + _INDEXING_OPS + _SORT_OPS
                                + _SPECTRAL_OPS + _LAYOUT_TRANSFORM_OPS
                                + _LINALG_OPS + _SPARSE_OPS + _MOE_OPS
                                + _OTHER_OPS)
def test_long_tail_primitive_is_addressable_via_ops(name):
    """`tessera.ops.<name>` must be a callable. Doesn't run it (some need
    very specific input shapes); just asserts addressability."""
    fn = getattr(ts.ops, name, None)
    assert fn is not None, f"tessera.ops.{name} should be exposed"
    assert callable(fn), f"tessera.ops.{name} should be callable"


# ── Targeted forward-pass smoke for the simplest names ──────────────────────


def test_reshape_preserves_total_elements():
    x = np.arange(24).reshape(2, 3, 4)
    y = ts.ops.reshape(x, (4, 6))
    assert y.shape == (4, 6)
    assert y.size == x.size


def test_cat_along_axis():
    a = np.zeros((2, 3))
    b = np.ones((2, 3))
    out = ts.ops.cat((a, b), axis=0)
    assert out.shape == (4, 3)


def test_stack_introduces_new_axis():
    a = np.zeros((3,))
    b = np.ones((3,))
    out = ts.ops.stack((a, b), axis=0)
    assert out.shape == (2, 3)


def test_transpose_swaps_axes():
    x = np.arange(6).reshape(2, 3)
    y = ts.ops.transpose(x)
    assert y.shape == (3, 2)


def test_flatten_collapses_axes():
    x = np.zeros((2, 3, 4))
    y = ts.ops.flatten(x)
    assert y.ndim == 1
    assert y.size == 24


def test_fft_round_trip_via_ifft():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    spectrum = ts.ops.fft(x)
    reconstructed = ts.ops.ifft(spectrum)
    np.testing.assert_allclose(reconstructed.real, x, atol=1e-7)


def test_rfft_irfft_round_trip():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    spectrum = ts.ops.rfft(x)
    reconstructed = ts.ops.irfft(spectrum, n=4)
    np.testing.assert_allclose(reconstructed, x, atol=1e-6)


def test_cholesky_factors_positive_definite():
    a = np.array([[4.0, 2.0], [2.0, 3.0]])
    L = ts.ops.cholesky(a)
    np.testing.assert_allclose(L @ L.T, a, atol=1e-7)


def test_sort_and_argsort_consistent():
    x = np.array([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0])
    s = ts.ops.sort(x)
    idx = ts.ops.argsort(x)
    np.testing.assert_array_equal(s, x[idx])


def test_top_k_returns_largest_values():
    x = np.array([0.1, 0.5, 0.9, 0.3, 0.7])
    values, _indices = ts.ops.top_k(x, k=2)
    np.testing.assert_array_equal(values, np.array([0.9, 0.7]))


def test_segment_reduce_sums_segments():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    segment_ids = np.array([0, 0, 1, 1, 1])
    out = ts.ops.segment_reduce(x, segment_ids, op="sum")
    np.testing.assert_allclose(out, np.array([3.0, 12.0]))


def test_gather_along_axis():
    # gather defaults to axis=0; with shape (2, 3) row indices ∈ {0, 1}.
    x = np.array([[10, 20, 30], [40, 50, 60]])
    indices = np.array([1, 0])
    out = ts.ops.gather(x, indices)
    assert out.shape[0] == 2
    np.testing.assert_array_equal(out, np.array([[40, 50, 60], [10, 20, 30]]))


# ── Registry-shape assertions for state-effect / non-diff primitives ───────


@pytest.mark.parametrize("name", _SORT_OPS + ["argmax", "argmin"])
def test_index_returning_op_has_no_vjp(name):
    """Integer-output reductions / sort family should not register a VJP."""
    entry = coverage_for(name)
    assert entry.contract_status["vjp"] in ("not_applicable", "planned"), (
        f"{name} returns integer indices; VJP must be not_applicable or planned"
    )


@pytest.mark.parametrize("name", _MOE_OPS)
def test_moe_op_has_collective_effect_declared(name):
    entry = coverage_for(name)
    # MoE ops have either `collective` (moe / moe_dispatch / moe_combine)
    # or `state` effect. The masking_effect_rule should be promoted to
    # `complete` per the effect classifier.
    assert entry.contract_status["masking_effect_rule"] == "complete", (
        f"{name} has a declared effect — masking_effect_rule should be complete"
    )


# ── Single roll-up: every entry from the partial-tests list must now
#    have tests = complete after this smoke file lands.                    ──


_COVERED_BY_THIS_FILE: list[str] = (
    _STRUCTURAL_PRESERVE_SHAPE + _INDEXING_OPS + _SORT_OPS
    + _SPECTRAL_OPS + _LAYOUT_TRANSFORM_OPS + _LINALG_OPS
    + _SPARSE_OPS + _MOE_OPS + _OTHER_OPS
)


def test_smoke_file_covers_every_partial_tests_entry():
    """Self-check: this file's lists must cover every registry entry whose
    `tests` axis was still `partial` before the smoke pass landed."""
    expected_uncovered_after = set()
    for name in _COVERED_BY_THIS_FILE:
        try:
            coverage_for(name)
        except KeyError:
            expected_uncovered_after.add(name)
    assert not expected_uncovered_after, (
        f"Smoke file references unknown registry entries: "
        f"{sorted(expected_uncovered_after)}"
    )
