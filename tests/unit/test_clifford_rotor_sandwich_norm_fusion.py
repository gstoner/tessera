"""Close-out gap #6 — GA cross-op fusion: rotor_sandwich → norm.

The canonical rotor-invariant ``norm(rotor_sandwich(R, x))`` previously dispatched
two GA ops with an intermediate multivector round-trip. This suite locks the
fused path:

  * ``ga.rotor_sandwich_norm`` (and the IR fusion pass) is numerically identical
    to the unfused composition;
  * ``@clifford_jit`` collapses the chain to a single
    ``clifford_rotor_sandwich_norm`` op in the plan;
  * the fusion only fires when the intermediate sandwich is consumed once;
  * the structural ``lower_function_to_ir`` stays unfused (fusion is a separate
    decorator pass).
"""

from __future__ import annotations

import numpy as np
import pytest

from tessera import ga
from tessera import _apple_gpu_backend as agb
from tessera import _jit_boundary as jb
from tessera.ga.multivector import Multivector
from tessera.ga.signature import Cl
from tessera.compiler import backend_manifest as bm
from tessera.compiler.clifford_jit import (
    clifford_jit, lower_function_to_ir, _fuse_rotor_sandwich_norm,
)

_GPU = agb.is_available() and jb.is_available()
gpu = pytest.mark.skipif(not _GPU, reason="apple_gpu runtime / libtessera_jit unavailable")


def _mv(seed, shape=(4, 8)):
    return Multivector(
        np.random.default_rng(seed).standard_normal(shape).astype(np.float32), Cl(3, 0, 0))


# ── surface + numerics ───────────────────────────────────────────────────────
def test_rotor_sandwich_norm_matches_unfused_reference():
    R, x = _mv(0), _mv(1)
    fused = np.asarray(ga.rotor_sandwich_norm(R, x))
    ref = np.asarray(ga.norm(ga.rotor_sandwich(R, x)))
    np.testing.assert_allclose(fused, ref, atol=1e-4)
    assert fused.shape == (4,)


def test_unbatched_returns_scalar():
    R = Multivector(np.random.default_rng(2).standard_normal(8).astype(np.float32), Cl(3, 0, 0))
    x = Multivector(np.random.default_rng(3).standard_normal(8).astype(np.float32), Cl(3, 0, 0))
    fused = np.asarray(ga.rotor_sandwich_norm(R, x))
    ref = np.asarray(ga.norm(ga.rotor_sandwich(R, x)))
    np.testing.assert_allclose(fused, ref, atol=1e-4)


# ── IR fusion pass ───────────────────────────────────────────────────────────
def test_fusion_pass_collapses_chain():
    def f(rotor, points):
        return ga.norm(ga.rotor_sandwich(rotor, points))

    structural = lower_function_to_ir(f)
    assert [o.op_name for o in structural.ops] == [
        "clifford_rotor_sandwich", "clifford_norm"]  # lowering stays unfused
    fused = _fuse_rotor_sandwich_norm(structural)
    assert [o.op_name for o in fused.ops] == ["clifford_rotor_sandwich_norm"]
    assert fused.ops[0].operand_refs == ("rotor", "points")
    assert fused.return_ref == fused.ops[0].result_name


def test_fusion_skips_when_intermediate_reused():
    # The sandwich result feeds both norm and another op → cannot fuse.
    def f(rotor, points):
        s = ga.rotor_sandwich(rotor, points)
        return ga.geometric_product(s, ga.grade_projection(s, 2))

    ir = _fuse_rotor_sandwich_norm(lower_function_to_ir(f))
    assert "clifford_rotor_sandwich_norm" not in [o.op_name for o in ir.ops]


def test_fusion_noop_for_plain_norm():
    def f(a):
        return ga.norm(a)

    ir = _fuse_rotor_sandwich_norm(lower_function_to_ir(f))
    assert [o.op_name for o in ir.ops] == ["clifford_norm"]


# ── manifest ─────────────────────────────────────────────────────────────────
def test_fusion_op_in_manifest_not_primitives():
    assert "clifford_rotor_sandwich_norm" in bm._CLIFFORD_APPLE_GPU_FUSED
    assert "clifford_rotor_sandwich_norm" in bm._CLIFFORD_FUSION_OPS
    assert "clifford_rotor_sandwich_norm" not in bm._CLIFFORD_PRIMITIVES


# ── @clifford_jit end-to-end ─────────────────────────────────────────────────
def test_clifford_jit_plan_is_fused():
    @clifford_jit(target="apple_gpu")
    def f(rotor, points):
        return ga.norm(ga.rotor_sandwich(rotor, points))

    assert f.artifact.op_names() == ("clifford_rotor_sandwich_norm",)


@gpu
def test_clifford_jit_fused_executes_correctly():
    @clifford_jit(target="apple_gpu")
    def f(rotor, points):
        return ga.norm(ga.rotor_sandwich(rotor, points))

    R, x = _mv(7), _mv(8)
    out = np.asarray(f(R, x))
    ref = np.asarray(ga.norm(ga.rotor_sandwich(R, x)))
    np.testing.assert_allclose(out, ref, atol=1e-4)
    # One fused dispatch, not two.
    assert f.plan_matches_routes()
    assert len(f.last_routes()) == 1
