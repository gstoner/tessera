"""An op's dtype set is derived from its numeric policy, not a blanket default.

`_ops()` defaulted every op a target enumerated to ``("fp32", "f32")``. That was
not conservative, it was wrong and self-inconsistent::

    gelu(bf16)   cpu=n  x86=n  apple_cpu=n  sm90=n  gfx942=n  |  gfx1151=Y

`gfx1151` said yes only because it declares ONE op, so everything else fell
through to the target's dtype tuple; `gfx942` declares four and got the blanket
default. The same op, opposite answers, decided by how many rows a target
happened to enumerate — and CDNA3 is not worse at bf16 than RDNA 3.5.

It also contradicted the reduced-precision work directly: the storage-dtype
wrapper computes `gelu(bf16)` at f32 and stores back, while this gate said the
backend could not do it.

The derivation: an op admits a target storage dtype when the op's declared
ACCUMULATOR can hold it. That is what makes "use f64 where the algorithm expects
it" concrete — the decomposition kinds declare `accum="fp64"` and admit fp64,
while an f32-accumulating activation does not.
"""

from __future__ import annotations

import pytest

from tessera.compiler.capabilities import (
    TARGET_CAPABILITIES,
    get_target_capability,
    supports_op,
)
from tessera.compiler.op_catalog import _SPECS
from tessera.compiler.primitive_coverage import (
    LOWERING_NUMERIC_POLICY,
    _policy_for_name,
)

_ALL_TARGETS = ("cpu", "x86", "apple_cpu", "apple_gpu", "nvidia_sm90",
                "nvidia_sm120", "rocm_gfx942", "rocm_gfx1151")


def test_every_catalog_op_has_a_numeric_policy():
    """Coverage went 59/313 → 313/313 via lowering-kind defaults.

    Writing 254 more per-op entries would have been the wrong shape: an
    accumulator is a property of what the op DOES, and the catalog already
    groups ops by exactly that.
    """
    missing = sorted({
        s.public_name for s in _SPECS if _policy_for_name(s.public_name) is None
    })
    assert not missing, f"catalog ops with no numeric policy: {missing[:20]}"


def test_every_lowering_kind_maps_to_a_policy():
    """A kind with no policy silently falls back to the blanket default."""
    kinds = {s.lowering for s in _SPECS}
    unmapped = sorted(kinds - set(LOWERING_NUMERIC_POLICY))
    assert not unmapped, f"lowering kinds with no numeric policy: {unmapped}"


@pytest.mark.parametrize("target", _ALL_TARGETS)
def test_bf16_activation_is_supported_wherever_the_target_declares_bf16(target):
    """The headline fix, and the one that removes the contradiction."""
    declared = get_target_capability(target).supported_dtypes
    if "bf16" not in declared:
        pytest.skip(f"{target} does not declare bf16 storage")
    for op in ("tessera.gelu", "tessera.softmax", "tessera.silu"):
        assert supports_op(target, op, dtype="bf16").supported, (
            f"{target} rejects {op} at bf16 while declaring bf16 storage"
        )


def test_all_targets_agree_about_bf16_gelu():
    """Consistency is the point: the old answer varied by declaration
    bookkeeping rather than by architecture."""
    answers = {
        t: supports_op(t, "tessera.gelu", dtype="bf16").supported
        for t in _ALL_TARGETS
        if "bf16" in get_target_capability(t).supported_dtypes
    }
    assert len(set(answers.values())) == 1, f"targets disagree: {answers}"


@pytest.mark.parametrize("target", ("x86", "nvidia_sm120", "rocm_gfx942"))
def test_f64_is_admitted_exactly_where_the_algorithm_declares_it(target):
    """Decompositions declare `accum="fp64"`; activations do not.

    This is "it is ok to use f64 when the core algorithm expects it" as a
    checkable rule rather than a case-by-case judgement.
    """
    if "fp64" not in get_target_capability(target).supported_dtypes:
        pytest.skip(f"{target} does not declare fp64")
    for op in ("tessera.cholesky", "tessera.qr", "tessera.svd"):
        assert supports_op(target, op, dtype="fp64").supported, op
    for op in ("tessera.gelu", "tessera.softmax"):
        assert not supports_op(target, op, dtype="fp64").supported, (
            f"{op} claims fp64 while accumulating at f32"
        )


def test_explicit_backend_declarations_still_win():
    """A backend that PROVED a set states it, and derivation must not touch it.

    sm120's matmul carries 13 dtypes including int8/int4/nvfp4 — integer and
    sub-byte support an f32 accumulator says nothing about.
    """
    cap = get_target_capability("nvidia_sm120").supported_ops["tessera.matmul"]
    assert not cap.dtypes_derived, "sm120 matmul must keep its explicit tuple"
    for dtype in ("nvfp4", "fp8_e4m3", "int8", "fp64"):
        assert dtype in cap.dtypes, f"sm120 matmul lost {dtype}"


def test_derived_dtypes_are_marked_and_do_not_become_kernel_claims():
    """LEGALITY and KERNEL EXISTENCE are different claims.

    sm90 declares fp8 storage and `gelu` accumulates at f32, so `gelu(fp8)` is
    EXPRESSIBLE — but no fp8 gelu kernel exists. The backend manifest is a
    kernel claim, so derived dtypes must not reach it. Without this the sm90
    dashboard asserted fp8 kernels for every activation, the same over-claim
    that once put nine phantom FFT rows on the NVIDIA dashboard.
    """
    from tessera.compiler.backend_manifest import _capability_status

    cap = get_target_capability("nvidia_sm90").supported_ops["tessera.gelu"]
    assert cap.dtypes_derived, "gelu on sm90 should be derived, not declared"
    assert "fp8_e4m3" in cap.dtypes, "legality should admit fp8 storage"

    entry = _capability_status("nvidia_sm90", "gelu")
    if entry is not None:
        _status, manifest_dtypes = entry
        assert "fp8_e4m3" not in manifest_dtypes, (
            f"manifest claims an fp8 gelu kernel on sm90: {manifest_dtypes}"
        )


def test_no_target_admits_a_dtype_it_does_not_declare():
    """Derivation intersects with the target; it never widens past it."""
    for name, target in TARGET_CAPABILITIES.items():
        declared = set(target.supported_dtypes)
        if not declared:
            continue
        for op_name, cap in target.supported_ops.items():
            if not cap.dtypes_derived:
                continue
            # Complex is deliberately NOT in any target's storage tuple -- it
            # is a logical dtype carried in an interleaved real pair, declared
            # on the transform ops that carry it. Excluding it here is the
            # design, not an exemption.
            extra = {
                d for d in cap.dtypes
                if d not in declared and d not in ("fp32", "f32")
                and "complex" not in d
            }
            assert not extra, (
                f"{name}/{op_name} derived {sorted(extra)} which the target "
                f"does not declare"
            )
