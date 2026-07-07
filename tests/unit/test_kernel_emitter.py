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

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
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
    KernelRunner,
    KernelSource,
    METAL_TARGETS,
    RunnerError,
    SpecPolicy,
    active_runner,
    bucket_key,
    emit_kernel,
    get_emitter,
    get_runner,
    register_emitter,
    register_runner,
)

_REPO = Path(__file__).resolve().parents[2]


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
    # A genuinely unregistered target (x86/rocm now register real emitters).
    with pytest.raises(EmitError, match="no KernelEmitter registered"):
        emit_kernel(F.FusedRegion(epilogue=("relu",)), "no_such_backend")


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


def test_apple_emitter_rejects_dynamic_spec():
    # DYNAMIC isn't implemented yet — the emitter must refuse, not emit the
    # bucket body mislabeled as dynamic (Decision #21).
    region = F.FusedRegion(epilogue=("relu",))
    with pytest.raises(EmitError, match="does not yet support SpecPolicy.DYNAMIC"):
        emit_kernel(region, "apple_gpu", SpecPolicy.DYNAMIC)


# ---- B2c: symbolic dims + SpecPolicy shape bucketing ------------------------

def test_bucket_key_per_policy():
    assert bucket_key(None, SpecPolicy.BUCKET) is None            # shape-anonymous
    assert bucket_key((8, 12, 16), SpecPolicy.STATIC) == (8, 12, 16)   # exact
    assert bucket_key((7, 13, 30), SpecPolicy.BUCKET) == (8, 16, 32)   # next pow2
    # DYNAMIC keys on symbolic identity, not values
    assert bucket_key((7, 13), SpecPolicy.DYNAMIC, dim_names=("m", "n")) == ("m", "n")
    assert bucket_key((7,), SpecPolicy.DYNAMIC) == ()            # no names -> all shapes


def test_bucket_key_dynamic_symbolic_survives_missing_dims():
    # AOT symbolic emit: no concrete example dims, but the symbolic identity must
    # still key the kernel — DYNAMIC is handled before the dims-is-None early
    # return, else distinct dynamic kernels collapse to one anonymous key.
    assert bucket_key(None, SpecPolicy.DYNAMIC, dim_names=("m", "n")) == ("m", "n")
    assert bucket_key(None, SpecPolicy.DYNAMIC) == ()
    # STATIC/BUCKET still go shape-anonymous without dims (can't bucket values).
    assert bucket_key(None, SpecPolicy.STATIC) is None
    assert bucket_key(None, SpecPolicy.BUCKET) is None


def test_bucket_key_matches_apple_shape_bucket_convention():
    # the emitter's bucketing must agree with the autotune corpus bucketing.
    from tessera.compiler.emit.apple_msl import _shape_bucket
    for n in (1, 2, 3, 31, 64, 65, 1000):
        assert bucket_key((n,), SpecPolicy.BUCKET) == (_shape_bucket(n),)


def test_emit_records_shape_key_per_policy():
    region = F.FusedRegion(epilogue=("gelu",))
    assert emit_kernel(region, "apple_gpu", SpecPolicy.STATIC, dims=(8, 12, 16)).shape_key == (8, 12, 16)
    assert emit_kernel(region, "apple_gpu", SpecPolicy.BUCKET, dims=(7, 13, 30)).shape_key == (8, 16, 32)
    assert emit_kernel(region, "apple_gpu", SpecPolicy.STATIC).shape_key is None   # no dims


def test_shape_key_is_metadata_not_codegen():
    # the specialization key must not change the emitted source — bucketing is a
    # cache/arbiter concern, not a codegen one.
    region = F.FusedRegion(epilogue=("gelu",))
    a = emit_kernel(region, "apple_gpu", SpecPolicy.STATIC, dims=(8, 12, 16))
    b = emit_kernel(region, "apple_gpu", SpecPolicy.BUCKET, dims=(64, 64, 64))
    assert a.source == b.source


def test_region_carries_symbolic_dim_names_non_semantically():
    # the dim_names carrier is Graph-IR bookkeeping — present or absent, the
    # synthesized source is identical (the synth path uses concrete operand dims).
    plain = F.FusedRegion(epilogue=("gelu",))
    symbolic = F.FusedRegion(epilogue=("gelu",), dim_names=("batch", "d_ff", "d_model"))
    assert symbolic.dim_names == ("batch", "d_ff", "d_model")
    assert plain.dim_names is None
    assert (synthesize_matmul_epilogue_msl(symbolic)
            == synthesize_matmul_epilogue_msl(plain))


def test_get_emitter_bootstraps_apple_without_prior_import():
    # Cold path: reach the registry via the public API without importing the
    # facade / apple_msl first. get_emitter must bootstrap the Apple reference
    # emitter so "apple_gpu" is available regardless of import order.
    code = (
        "from tessera.compiler.emit.kernel_emitter import emit_kernel, SpecPolicy\n"
        "from tessera.compiler.fusion_core import FusedRegion\n"
        "import tessera.compiler.emit.kernel_emitter as ke\n"
        "assert 'apple_gpu' not in ke._EMITTERS, 'apple must not be pre-registered'\n"
        "ks = emit_kernel(FusedRegion(epilogue=('relu',)), 'apple_gpu', SpecPolicy.STATIC)\n"
        "assert ks.lang == 'msl'\n"
        "print('BOOTSTRAP_OK')\n"
    )
    r = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True,
        cwd=str(_REPO), env={**os.environ, "PYTHONPATH": "python"})
    assert r.returncode == 0, r.stderr
    assert "BOOTSTRAP_OK" in r.stdout


# ---- B2b: injected KernelRunner ---------------------------------------------

def test_apple_runner_registered_and_active():
    # importing the facade imported apple_msl, which self-registered its runner.
    assert active_runner() is not None
    assert get_runner("apple_gpu").target == "apple_gpu"


def test_register_runner_requires_target():
    class _Bad(KernelRunner):
        target = ""
        def run_fused_region(self, region, *a, **k): ...
        def run_fused_attention(self, region, *a, **k): ...
        def run_gated_matmul_region(self, region, *a, **k): ...
        def run_pointwise_graph(self, region, *a, **k): ...
    with pytest.raises(ValueError, match="non-empty backend id"):
        register_runner(_Bad())


def test_unknown_runner_target_diagnostic():
    with pytest.raises(RunnerError, match="no KernelRunner registered"):
        get_runner("nvidia")


def test_oracle_routes_through_the_injected_runner():
    # The F4 oracle must call whatever runner is registered as active — proving
    # B2b injection replaced B1's hard-wired apple_msl import. Register a spy as
    # the default, confirm the core bridge dispatches to it, then restore the
    # global registry state (plain assignment — NOT monkeypatch, which would
    # record and re-apply the spy value on teardown and leak it downstream).
    import tessera.compiler.emit.kernel_emitter as ke
    import tessera.compiler.fusion_core as core

    calls: list[str] = []

    class _Spy(KernelRunner):
        target = "spy_backend"
        def run_fused_region(self, region, *a, **k):
            calls.append("region")
            return np.zeros((8, 16), "float32"), "reference"
        def run_fused_attention(self, region, *a, **k): ...
        def run_gated_matmul_region(self, region, *a, **k): ...
        def run_pointwise_graph(self, region, *a, **k): ...

    saved = ke._DEFAULT_RUNNER_TARGET
    try:
        register_runner(_Spy(), default=True)
        assert active_runner().target == "spy_backend"
        # _runner() (what an oracle resolves when no explicit runner is passed)
        # now dispatches to the registered active runner — the spy.
        _out, ex = core._runner().run_fused_region(F.FusedRegion(epilogue=("relu",)),
                                                   np.ones((8, 12), "float32"),
                                                   np.ones((12, 16), "float32"))
        assert calls == ["region"] and ex == "reference"
    finally:
        ke._DEFAULT_RUNNER_TARGET = saved
        ke._RUNNERS.pop("spy_backend", None)


# ---- B3: F4 oracle as a universal, per-backend correctness gate -------------

class _WrongRunner(KernelRunner):
    """A runner whose 'device' kernel diverges from the numpy reference."""
    target = "wrong_backend"
    def run_fused_region(self, region, A, B, bias=None, *a, **k):
        return np.full((A.shape[0], B.shape[1]), 999.0, np.float32), "metal_runtime"
    def run_fused_attention(self, region, *a, **k): ...
    def run_gated_matmul_region(self, region, *a, **k): ...
    def run_pointwise_graph(self, region, *a, **k): ...


class _RefRunner(_WrongRunner):
    """A runner whose kernel matches the numpy reference exactly."""
    target = "ref_backend"
    def run_fused_region(self, region, A, B, bias=None, *a, **k):
        return region.reference(A, B, bias), "metal_runtime"


def test_verify_gates_the_explicitly_injected_runner():
    # B3: the same numpy-reference oracle gates ANY backend's kernel via an
    # explicit `runner=` — no registry mutation, no Metal. A diverging runner is
    # rejected (False), a faithful one is trusted (True).
    F.clear_verification_cache()
    region = F.FusedRegion(epilogue=("relu",))
    assert F.verify_synthesized_region(region, runner=_WrongRunner(), force=True) is False
    assert F.verify_synthesized_region(region, runner=_RefRunner(), force=True) is True


def test_verify_default_runner_unchanged_without_injection():
    # Omitting runner keeps today's behavior: resolve the registered active
    # runner (Apple). A correct synthesizer verifies True (or True when no Metal).
    F.clear_verification_cache()
    assert F.verify_synthesized_region(F.FusedRegion(epilogue=("relu",))) is True


class _X86Wrong(KernelRunner):
    """A non-Metal backend whose kernel diverges — returns its OWN real tag."""
    target = "x86"
    def run_fused_region(self, region, A, B, bias=None, *a, **k):
        return np.full((A.shape[0], B.shape[1]), 999.0, np.float32), "x86_native"
    def run_fused_attention(self, region, *a, **k): ...
    def run_gated_matmul_region(self, region, *a, **k): ...
    def run_pointwise_graph(self, region, *a, **k): ...


def test_oracle_gates_non_metal_backends():
    # The F4 gate is backend-agnostic: a runner returning its own real-execution
    # tag ("x86_native", not "metal_runtime") is still compared to the reference,
    # so a non-Apple backend is genuinely gated — not trusted by default. A
    # reference/fallback tag (no real kernel) is still trusted.
    F.clear_verification_cache()
    region = F.FusedRegion(epilogue=("relu",))

    class _X86Ref(_X86Wrong):
        def run_fused_region(self, region, A, B, bias=None, *a, **k):
            return region.reference(A, B, bias), "x86_native"

    class _X86Fallback(_X86Wrong):
        def run_fused_region(self, region, A, B, bias=None, *a, **k):
            return None, "fallback"

    assert F.verify_synthesized_region(region, runner=_X86Wrong(), force=True) is False
    assert F.verify_synthesized_region(region, runner=_X86Ref(), force=True) is True
    assert F.verify_synthesized_region(region, runner=_X86Fallback(), force=True) is True


def test_verify_cache_is_keyed_per_runner():
    # B3 P1: a verdict from one runner must NOT be reused for another, even
    # without force — the cache key includes the backend identity. Verify a
    # faithful runner (True), then WITHOUT force a wrong runner must re-execute
    # (distinct key) and be rejected (False), not return the cached True.
    F.clear_verification_cache()
    region = F.FusedRegion(epilogue=("relu",))
    assert F.verify_synthesized_region(region, runner=_RefRunner()) is True
    assert F.verify_synthesized_region(region, runner=_WrongRunner()) is False
    # and the reverse order: wrong cached False must not block the faithful one
    F.clear_verification_cache()
    assert F.verify_synthesized_region(region, runner=_WrongRunner()) is False
    assert F.verify_synthesized_region(region, runner=_RefRunner()) is True
