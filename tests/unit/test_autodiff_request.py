"""Phase 1 — the public differentiation request + backward provenance facet.

Locks the decoration-time contract: `@jit(autodiff="reverse", wrt=...)` validates
inputs, emits the `tessera.autodiff` intent into the Graph IR module, and resolves
a typed backward facet that never over-claims native execution.
"""

from __future__ import annotations

import pytest

import tessera
from tessera.compiler import autodiff_request as ad
from tessera.compiler.autodiff_request import BackwardStatus, TesseraAutodiffError


# ── pure request validation ────────────────────────────────────────────────

def _fn(x, w):  # a stand-in with an introspectable signature
    return x


def test_no_autodiff_is_none():
    assert ad.build_request(_fn, autodiff=None, wrt=None) is None


def test_reverse_defaults_wrt_to_all_params():
    req = ad.build_request(_fn, autodiff="reverse", wrt=None)
    assert req is not None
    assert req.mode == "reverse"
    assert req.wrt == ("x", "w")


def test_reverse_with_explicit_wrt_subset():
    req = ad.build_request(_fn, autodiff="reverse", wrt=("w",))
    assert req.wrt == ("w",)


def test_wrt_name_not_in_signature_raises():
    with pytest.raises(TesseraAutodiffError, match="not in the function's parameters"):
        ad.build_request(_fn, autodiff="reverse", wrt=("nope",))


def test_forward_and_jvp_modes_are_canonicalized():
    forward = ad.build_request(_fn, autodiff="forward", wrt=("w",))
    alias = ad.build_request(_fn, autodiff="jvp", wrt=("w",))
    assert forward == alias
    assert forward.mode == "forward"
    assert forward.wrt_indices == (1,)


def test_unknown_mode_rejected():
    with pytest.raises(TesseraAutodiffError, match="unknown"):
        ad.build_request(_fn, autodiff="sideways", wrt=None)


def test_wrt_without_autodiff_rejected():
    with pytest.raises(TesseraAutodiffError, match="without"):
        ad.build_request(_fn, autodiff=None, wrt=("x",))


def test_empty_wrt_rejected():
    with pytest.raises(TesseraAutodiffError, match="empty"):
        ad.build_request(_fn, autodiff="reverse", wrt=())


def test_duplicate_wrt_rejected_before_compiler_abi():
    with pytest.raises(TesseraAutodiffError, match="duplicate"):
        ad.build_request(_fn, autodiff="forward", wrt=("x", "x"))


# ── intent attributes ──────────────────────────────────────────────────────

def test_module_intent_attrs_are_valid_mlir_strings():
    req = ad.build_request(_fn, autodiff="reverse", wrt=("x", "w"))
    attrs = req.module_intent_attrs()
    assert attrs["tessera.autodiff"] == '"reverse"'
    assert attrs["tessera.autodiff.wrt"] == '["x", "w"]'
    assert attrs["tessera.autodiff.wrt_indices"] == "[0, 1]"


# ── backward provenance resolution ─────────────────────────────────────────

def test_provenance_not_requested_when_no_request():
    prov = ad.resolve_backward_provenance(None)
    assert prov.status is BackwardStatus.NOT_REQUESTED
    assert not prov.requested


def test_provenance_ir_transformed_by_default_today():
    req = ad.build_request(_fn, autodiff="reverse", wrt=None)
    prov = ad.resolve_backward_provenance(req)
    assert prov.status is BackwardStatus.IR_TRANSFORMED
    assert not prov.native  # never over-claims native execution today


def test_provenance_native_required_needs_exact_target_proof():
    req = ad.build_request(_fn, autodiff="reverse", wrt=None, native_required=True)
    prov = ad.resolve_backward_provenance(req, target="cpu")
    assert prov.status is BackwardStatus.UNSUPPORTED
    assert "native_required" in prov.reason
    assert "exact-target device verification" in prov.reason


def test_provenance_native_when_backend_wired():
    """Phase 4 hook: when a native backward path exists, the facet flips."""
    req = ad.build_request(_fn, autodiff="reverse", wrt=None)
    prov = ad.resolve_backward_provenance(req, has_native_backward=True)
    assert prov.status is BackwardStatus.NATIVE_EXECUTABLE
    assert prov.native


def test_provenance_native_sourced_from_matrix_rocm_flash_attn():
    """Phase 4 (A3): with op_families + target, the native hook is sourced from
    the execution matrix — ROCm flash_attn has a device-proven backward, so the
    facet resolves NATIVE_EXECUTABLE without an explicit bool."""
    req = ad.build_request(_fn, autodiff="reverse", wrt=None)
    prov = ad.resolve_backward_provenance(
        req, target="rocm", op_families=("flash_attn",))
    assert prov.status is BackwardStatus.NATIVE_EXECUTABLE
    assert prov.native


def test_provenance_native_required_honored_on_rocm_flash_attn():
    """native_required=True is honored (not UNSUPPORTED) where the matrix proves
    a native backward — the first place enforcement pays off (Phase 4 A3)."""
    req = ad.build_request(
        _fn, autodiff="reverse", wrt=None, native_required=True)
    prov = ad.resolve_backward_provenance(
        req, target="rocm", op_families=("flash_attn",))
    assert prov.status is BackwardStatus.NATIVE_EXECUTABLE


def test_provenance_mixed_program_not_native():
    """A program is native only if EVERY component op has a native backward.
    flash_attn+softmax on rocm: softmax has no backward launch → not native."""
    req = ad.build_request(_fn, autodiff="reverse", wrt=None)
    prov = ad.resolve_backward_provenance(
        req, target="rocm", op_families=("flash_attn", "softmax"))
    assert prov.status is BackwardStatus.IR_TRANSFORMED
    assert not prov.native


def test_provenance_native_sourced_false_off_target():
    """Same flash_attn family on cpu (no native backward row) stays honest."""
    req = ad.build_request(_fn, autodiff="reverse", wrt=None)
    prov = ad.resolve_backward_provenance(
        req, target="cpu", op_families=("flash_attn",))
    assert prov.status is BackwardStatus.IR_TRANSFORMED
    assert not prov.native


def test_forward_provenance_is_ir_transformed_not_backward_native():
    req = ad.build_request(_fn, autodiff="forward", wrt=("w",))
    prov = ad.resolve_differentiation_provenance(
        req, target="x86", op_families=("matmul",)
    )
    assert prov.status is ad.DifferentiationStatus.IR_TRANSFORMED
    assert prov.mode == "forward"
    assert not prov.native
    with pytest.raises(ValueError, match="reverse-mode"):
        ad.resolve_backward_provenance(req)


def test_forward_native_required_fails_closed_without_target_jvp_proof():
    req = ad.build_request(
        _fn, autodiff="forward", wrt=("w",), native_required=True
    )
    prov = ad.resolve_differentiation_provenance(req, target="rocm")
    assert prov.status is ad.DifferentiationStatus.UNSUPPORTED
    assert "no native JVP execution path" in prov.reason


def test_forward_provenance_rejects_family_without_tangent_interface():
    req = ad.build_request(_fn, autodiff="forward", wrt=("x",))
    prov = ad.resolve_differentiation_provenance(
        req, target="cpu", op_families=("sin",)
    )
    assert prov.status is ad.DifferentiationStatus.UNSUPPORTED
    assert "no TangentInterface" in prov.reason


# ── @jit integration ───────────────────────────────────────────────────────

@tessera.jit(autodiff="reverse", wrt=("x", "w"))
def _loss(x, w):
    return tessera.ops.matmul(x, w)


def test_jit_exposes_request_and_provenance():
    assert _loss.differentiation_request.mode == "reverse"
    assert _loss.differentiation_request.wrt == ("x", "w")
    assert _loss.backward_provenance.status is BackwardStatus.IR_TRANSFORMED


def test_jit_emits_intent_into_graph_ir_module():
    attrs = _loss.graph_ir.module_attrs
    assert attrs.get("tessera.autodiff") == '"reverse"'
    assert attrs.get("tessera.autodiff.wrt") == '["x", "w"]'


def test_jit_emits_intent_on_the_function_where_the_pass_reads_it():
    # The C++ --tessera-autodiff pass keys off the func.func attribute, not the
    # module — so the marker must land in the function's fn_attrs.
    fns = _loss.graph_ir.functions
    assert fns, "expected at least one emitted function"
    target = next((f for f in fns if f.name == "_loss"), fns[0])
    assert target.fn_attrs.get("tessera.autodiff") == '"reverse"'


def test_jit_mirrors_backward_facet_onto_compile_result():
    cr = _loss.compile_result
    if cr is not None:  # eager path may not build a full CompileResult
        assert cr.backward is not None
        assert cr.backward.status is BackwardStatus.IR_TRANSFORMED


@tessera.jit(autodiff="reverse", wrt=("q", "k", "v"), target="rocm")
def _fa_rocm(q, k, v):
    return tessera.ops.flash_attn(q, k, v)


def test_jit_rocm_flash_attn_backward_is_native_executable():
    """Phase 4 (A3) end-to-end: @jit(autodiff="reverse", target="rocm") over
    flash_attn resolves its backward facet to NATIVE_EXECUTABLE via the
    matrix-sourced hook (component_ops → execution_matrix.has_native_backward) —
    the first native-backward training path in the compiler."""
    assert _fa_rocm.compile_result is not None
    assert _fa_rocm.compile_result.component_ops == ("flash_attn",)
    assert _fa_rocm.backward_provenance.status is BackwardStatus.NATIVE_EXECUTABLE
    assert _fa_rocm.backward_provenance.native


def test_jit_without_autodiff_has_no_request():
    @tessera.jit
    def plain(x, w):
        return tessera.ops.matmul(x, w)
    assert plain.differentiation_request is None
    assert plain.backward_provenance.status is BackwardStatus.NOT_REQUESTED
    assert "tessera.autodiff" not in plain.graph_ir.module_attrs


@tessera.jit(autodiff="jvp", wrt=("w",))
def _forward_loss(x, w):
    return tessera.ops.matmul(x, w)


def test_jit_exposes_forward_request_and_mode_neutral_provenance():
    assert _forward_loss.differentiation_request.mode == "forward"
    assert _forward_loss.differentiation_request.wrt_indices == (1,)
    assert (
        _forward_loss.differentiation_provenance.status
        is ad.DifferentiationStatus.IR_TRANSFORMED
    )
    assert _forward_loss.backward_provenance.status is BackwardStatus.NOT_REQUESTED
    assert _forward_loss.graph_ir.module_attrs["tessera.autodiff"] == '"forward"'
    assert _forward_loss.graph_ir.module_attrs["tessera.autodiff.wrt_indices"] == "[1]"
    if _forward_loss.compile_result is not None:
        assert _forward_loss.compile_result.differentiation.mode == "forward"
        assert _forward_loss.compile_result.backward is None


def test_jit_compiled_jvp_ir_uses_requested_tangent_abi():
    from tessera import _jit_boundary

    if _jit_boundary._find_tessera_opt() is None:
        pytest.skip("tessera-opt not built")
    import numpy as np

    ir = _forward_loss.compiled_jvp_ir(
        np.ones((2, 3), dtype=np.float32),
        np.ones((3, 4), dtype=np.float32),
    )
    assert "@_forward_loss__jvp" in ir
    signature = ir.split("func.func private @_forward_loss__jvp", 1)[1].split(
        "{", 1
    )[0]
    assert signature.count("tensor<2x3xf32>") == 1
    assert signature.count("tensor<3x4xf32>") == 2


@tessera.jit(autodiff="reverse", wrt=("x",))
def _exact_hvp_square(x):
    return tessera.ops.mul(x, x)


def test_jit_compiled_hvp_ir_composes_reverse_and_forward_exactly():
    from tessera import _jit_boundary

    if _jit_boundary._find_tessera_opt() is None:
        pytest.skip("tessera-opt not built")
    import numpy as np

    ir = _exact_hvp_square.compiled_hvp_ir(
        np.ones((4,), dtype=np.float32)
    )
    assert "@_exact_hvp_square__bwd__jvp" in ir
    assert "tessera.autodiff.hvp = @_exact_hvp_square__bwd__jvp" in ir
    assert "finite_difference" not in ir
