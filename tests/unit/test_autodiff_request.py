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


def test_forward_mode_is_rejected_as_planned():
    with pytest.raises(TesseraAutodiffError, match="reverse-mode only"):
        ad.build_request(_fn, autodiff="forward", wrt=None)


def test_unknown_mode_rejected():
    with pytest.raises(TesseraAutodiffError, match="unknown"):
        ad.build_request(_fn, autodiff="sideways", wrt=None)


def test_wrt_without_autodiff_rejected():
    with pytest.raises(TesseraAutodiffError, match="without"):
        ad.build_request(_fn, autodiff=None, wrt=("x",))


def test_empty_wrt_rejected():
    with pytest.raises(TesseraAutodiffError, match="empty"):
        ad.build_request(_fn, autodiff="reverse", wrt=())


# ── intent attributes ──────────────────────────────────────────────────────

def test_module_intent_attrs_are_valid_mlir_strings():
    req = ad.build_request(_fn, autodiff="reverse", wrt=("x", "w"))
    attrs = req.module_intent_attrs()
    assert attrs["tessera.autodiff"] == '"reverse"'
    assert attrs["tessera.autodiff.wrt"] == '["x", "w"]'


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


def test_provenance_native_required_is_unsupported_today():
    req = ad.build_request(_fn, autodiff="reverse", wrt=None, native_required=True)
    prov = ad.resolve_backward_provenance(req, target="cpu")
    assert prov.status is BackwardStatus.UNSUPPORTED
    assert "native_required" in prov.reason and "Phase 4" in prov.reason


def test_provenance_native_when_backend_wired():
    """Phase 4 hook: when a native backward path exists, the facet flips."""
    req = ad.build_request(_fn, autodiff="reverse", wrt=None)
    prov = ad.resolve_backward_provenance(req, has_native_backward=True)
    assert prov.status is BackwardStatus.NATIVE_EXECUTABLE
    assert prov.native


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


def test_jit_without_autodiff_has_no_request():
    @tessera.jit
    def plain(x, w):
        return tessera.ops.matmul(x, w)
    assert plain.differentiation_request is None
    assert plain.backward_provenance.status is BackwardStatus.NOT_REQUESTED
    assert "tessera.autodiff" not in plain.graph_ir.module_attrs
