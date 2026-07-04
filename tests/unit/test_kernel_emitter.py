"""B2a — the target-agnostic KernelEmitter protocol (COMPILER_REFACTOR_PLAN W-B2).

B1 split fusion.py into fusion_core (arch-agnostic) + emit.apple_msl (Metal). B2
lifts the emitter behind a plugin protocol so a non-Apple backend reuses the
whole synthesizer by implementing one interface. These tests lock:

* the vocab is target-parametric — ``EpilogueOp.emit(target)`` /
  ``ReductionOp.emit(target)`` replace the Metal-only ``.msl`` field, and an
  unknown target raises (Decision #21: no silent wrong-language emit);
* ``AppleMSLEmitter`` is a faithful *wrapper* of the ``synthesize_*_msl`` bodies
  (byte-identical source), not a reimplementation;
* the registry / ``emit_kernel(region, target, spec)`` entry point + its
  unknown-target diagnostic;
* ``SpecPolicy`` carries the ``static | bucket | dynamic`` policy so the
  interface is dynamic-ready from day one.
"""

from __future__ import annotations

import pytest

import tessera.compiler.fusion as F
from tessera.compiler.emit.apple_msl import (
    AppleMSLEmitter,
    _ENTRY,
    synthesize_matmul_epilogue_msl,
    synthesize_norm_chain_msl,
)
from tessera.compiler.emit.kernel_emitter import (
    EmitError,
    KernelEmitter,
    KernelSource,
    METAL_TARGETS,
    SpecPolicy,
    emit_kernel,
    get_emitter,
    register_emitter,
)


# ---- vocab is target-parametric (EpilogueOp/ReductionOp.emit) ---------------

def test_epilogue_emit_returns_metal_body_for_metal_targets():
    op = F.EPILOGUE_OPS["gelu"]
    body = op.emit("apple_gpu")
    assert "tanh" in body
    # every Metal alias resolves to the same body
    for t in METAL_TARGETS:
        assert op.emit(t) == body


def test_epilogue_emit_unknown_target_raises():
    with pytest.raises(ValueError, match="no kernel snippet"):
        F.EPILOGUE_OPS["relu"].emit("ptx")


def test_reduction_emit_returns_block_with_eps_placeholder():
    block = F.REDUCTION_OPS["rmsnorm"].emit("metal")
    assert "{eps}" in block and "scores" in block
    with pytest.raises(ValueError, match="no kernel snippet"):
        F.REDUCTION_OPS["softmax"].emit("amdgcn")


def test_msl_field_is_private_now():
    # The Metal-only public field is gone; access is via emit(target).
    op = F.EPILOGUE_OPS["relu"]
    assert not hasattr(op, "msl")
    assert hasattr(op, "emit")


# ---- AppleMSLEmitter wraps synth faithfully ---------------------------------

def test_apple_emitter_wraps_matmul_epilogue_byte_identical():
    region = F.FusedRegion(epilogue=("gelu",))
    ks = emit_kernel(region, "apple_gpu", SpecPolicy.STATIC)
    assert isinstance(ks, KernelSource)
    assert ks.lang == "msl" and ks.entry == _ENTRY
    assert ks.spec is SpecPolicy.STATIC
    # the emitter must NOT alter the synthesized source
    assert ks.source == synthesize_matmul_epilogue_msl(region, dtype="f32")


def test_apple_emitter_dispatches_norm_chain():
    region = F.NormChainRegion(norm="rmsnorm")
    ks = emit_kernel(region, "apple_gpu")
    assert ks.source == synthesize_norm_chain_msl(region, dtype="f32")
    assert ks.spec is SpecPolicy.BUCKET   # plan default: bucket-first


def test_apple_emitter_can_emit_predicate():
    e = AppleMSLEmitter()
    assert e.can_emit(F.FusedRegion(epilogue=("relu",)))
    assert not e.can_emit(object())


def test_apple_emitter_rejects_unknown_region():
    with pytest.raises(EmitError, match="cannot emit"):
        AppleMSLEmitter().emit(object())


# ---- registry + entry point -------------------------------------------------

def test_registry_resolves_apple_and_reports_unknown_target():
    assert get_emitter("apple_gpu").target == "apple_gpu"
    with pytest.raises(EmitError, match="no KernelEmitter registered"):
        emit_kernel(F.FusedRegion(epilogue=("relu",)), "rocm")


def test_register_emitter_requires_target():
    class _Bad(KernelEmitter):
        target = ""
        lang = "x"
        def can_emit(self, region):  # noqa: D401
            return False
        def emit(self, region, *, spec=SpecPolicy.BUCKET, dtype="f32"):
            raise NotImplementedError
    with pytest.raises(ValueError, match="non-empty backend id"):
        register_emitter(_Bad())


# ---- SpecPolicy is dynamic-ready --------------------------------------------

def test_spec_policy_has_static_bucket_dynamic():
    assert {p.value for p in SpecPolicy} == {"static", "bucket", "dynamic"}
