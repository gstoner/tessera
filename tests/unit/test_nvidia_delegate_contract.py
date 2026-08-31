"""The shipped NVIDIA GEMM as the first *declared* delegate.

`test_delegate_contract.py` proves the contract machinery works and agrees
with the C++ verifier. This file proves it is actually *used*: until now every
registered candidate returned `None` from `contract_for_candidate`, so the
whole delegation contract had zero live users and the arbiter selected a
Tier-3 hand-tuned kernel on tier priority alone -- no declared accuracy
budget, no determinism claim, no architecture claim, no coverage claim.

Measured on the fleet's sm_120 (RTX 5070) before this landed:

    registered: nvidia_mma_gemm_shipped(T3) nvidia_mma_gemm_emitted(T2)
                nvidia_tile_matmul_direct(T2) nvidia_tile_matmul_shared(T2)
    winner    : nvidia_mma_gemm_shipped
    contracts : all four -> None

These tests are host-free: they check what the delegate *declares*. The
device-side claims (that it is in fact the fastest, and that its declared
budget holds) are proven separately in `tests/device/nvidia/`, because a
declaration is not evidence.
"""
from __future__ import annotations

import pytest

from tessera.compiler import fusion_core
from tessera.compiler.emit import candidate as C
from tessera.compiler.emit.delegate_contract import (
    DelegateContract,
    DelegateContractError,
    DelegatedCandidate,
    contract_for_candidate,
)

import tessera.compiler.emit.nvidia_cuda as nvidia_cuda


SHIPPED = "nvidia_mma_gemm_shipped"


def _shipped():
    for c in C.candidates_for("nvidia", C.OP_MATMUL):
        if c.name == SHIPPED:
            return c
    pytest.fail(f"{SHIPPED} is not registered for (nvidia, matmul)")


# ── the delegate exists and is declared ──────────────────────────────────────

def test_the_shipped_gemm_declares_a_contract():
    """The regression this whole file exists for: before it, every NVIDIA
    candidate returned None here while still winning arbitration as Tier 3."""
    assert contract_for_candidate(_shipped()) is not None


def test_tier_and_budget_are_derived_not_hand_set():
    shipped = _shipped()
    contract = contract_for_candidate(shipped)
    assert shipped.tier == contract.arbiter_tier() == C.Tier.HAND_TUNED
    assert shipped.accuracy_atol == contract.arbiter_accuracy_atol()
    assert shipped.accuracy_rtol == contract.arbiter_accuracy_rtol()


def test_the_name_survived_the_conversion():
    """`name` is a cache/dispatch key, not a claim.

    The autotune corpus and the E3 `force` hatch key on this exact string and
    predate the contract, so deriving it from `callee` would silently
    invalidate every persisted verdict and break `force` with no error.
    """
    assert _shipped().name == SHIPPED


# ── the claims are the ones the kernel can actually support ──────────────────

def test_it_claims_determinism_and_the_kernel_supports_that():
    """Grounded in `aot/tessera_nvidia_mma_f16_sm120_v1.cu`: one warp owns each
    16x8 output tile and reduces K serially into four accumulator registers.
    No atomics, no split-K, no cross-block reduction -- so it is reproducible
    run to run and may be selected inside `@jit(deterministic=True)`.

    This is the claim that would silently defeat a user-facing guarantee if it
    were wrong, which is why it is asserted rather than assumed.
    """
    assert contract_for_candidate(_shipped()).is_deterministic()


def test_it_does_not_claim_to_absorb_an_epilogue():
    """The kernel is a bare GEMM. `whole_region` would assert an
    epilogue-fusing ability it does not have, and would let it win a
    comparison against a candidate that really does fuse."""
    contract = contract_for_candidate(_shipped())
    assert contract.covers == "root_only"
    assert not contract.serves_whole_region()


def test_it_claims_a_bounded_not_exact_result():
    """f16/bf16 operands with f32 accumulation do not reproduce the reference
    bit for bit, so `reference_exact` would be an overclaim."""
    contract = contract_for_candidate(_shipped())
    assert contract.accuracy == "tolerance_bounded"
    assert contract.tolerance is not None and contract.tolerance_rel is not None


def test_the_architecture_claim_is_the_envelope_not_the_aot_target():
    """`tessera_nvidia_gemm.cpp` NVRTC-compiles `--gpu-architecture=compute_%d%d`
    for the LIVE device; the sm_120 cubin is an AOT fast path, not the limit.
    The kernel needs only `mma.sync.aligned.m16n8k16`, which is sm_80+.

    Declaring `sm_120` here would have been wrong in both directions at once:
    under-claiming the parts it runs on, while over-claiming precision about
    the one it was built for.
    """
    assert contract_for_candidate(_shipped()).arch == "sm_80+"


def test_the_footprint_key_is_not_the_architecture_claim():
    """`mma_arch` keys the analytical footprint model, whose `_STATIC_ISAS`
    holds exactly one NVIDIA record. It is deliberately NOT the contract's
    architecture claim -- conflating them is how `sm_120` became a hardcoded
    class attribute on a kernel that runs on sm_80 and later.
    """
    shipped = _shipped()
    assert shipped.mma_arch == "sm_120"
    assert contract_for_candidate(shipped).arch != shipped.mma_arch


# ── the callee family: one candidate, two bound symbols ──────────────────────

def test_each_dtype_declares_the_symbol_it_actually_calls():
    shipped = _shipped()
    for dtype, expected in nvidia_cuda._SHIPPED_GEMM_CALLEES.items():
        region = fusion_core.MatmulRegion(dtype=dtype)
        assert shipped.contract_for(region).callee == expected


def test_the_declared_callees_match_the_runtime_symbol_table():
    """Drift gate. The contract names a C symbol; `runtime` resolves it with
    `getattr(lib, sym)`. A rename on one side only would make the delegate
    declare a callee it does not call -- precisely the Python-vs-IR drift the
    contract exists to catch, and invisible without this assertion.
    """
    from tessera import runtime as rt

    for dtype, declared in nvidia_cuda._SHIPPED_GEMM_CALLEES.items():
        assert rt._NVIDIA_GEMM_SYMBOLS[dtype] == declared


def test_the_timing_path_binds_the_same_kernel_as_the_execution_path():
    """The device-latency helper calls `<symbol>_device`. If that mapping
    drifted, the measured latency would describe a kernel the arbiter never
    runs -- a number that is worse than no number, because it looks like
    evidence.
    """
    from tessera import runtime as rt

    for dtype, execute in rt._NVIDIA_GEMM_SYMBOLS.items():
        assert rt._NVIDIA_GEMM_DEVICE_SYMBOLS[dtype] == execute + "_device"


def test_every_served_dtype_has_a_declared_callee():
    """A dtype in `_GEMM_DTYPES` with no declared callee would fall back to the
    representative contract and claim the wrong symbol."""
    assert set(nvidia_cuda._GEMM_DTYPES) == set(nvidia_cuda._SHIPPED_GEMM_CALLEES)
    shipped = _shipped()
    assert set(shipped.contract_variants) == set(nvidia_cuda._GEMM_DTYPES)


# ── the family invariant ─────────────────────────────────────────────────────

def _contract(**overrides) -> DelegateContract:
    base = dict(callee="k", binding="c_abi", provenance="handwritten_kernel",
                arch="sm_80+", accuracy="tolerance_bounded", tolerance=1e-3,
                determinism="deterministic", covers="root_only")
    base.update(overrides)
    return DelegateContract(**base)


class _Delegate(DelegatedCandidate):
    def run(self, region, *inputs, **kwargs):  # pragma: no cover - unused
        raise NotImplementedError


@pytest.mark.parametrize("field,value", [
    ("arch", "sm_90"),
    ("determinism", "nondeterministic"),
    ("covers", "whole_region"),
    ("provenance", "vendor_library"),
    ("binding", "cuda_kernel"),
    ("accuracy", "reference_exact"),
])
def test_a_variant_may_not_contradict_the_family(field, value):
    """These fields describe the delegate, not one dtype route. A family whose
    members disagree on them has no honest representative contract -- and
    `contract_for_candidate` returns exactly that representative, so the
    disagreement would be invisible to every consumer that does not ask per
    dtype.
    """
    extra = {}
    if value == "reference_exact":
        extra = {"tolerance": None}
    with pytest.raises(DelegateContractError, match=field):
        _Delegate(_contract(), target="nvidia", op=C.OP_MATMUL,
                  variants={"float16": _contract(**{field: value}, **extra)})


def test_a_variant_may_differ_on_callee_and_tolerance():
    """The two fields that legitimately vary per dtype: which symbol is bound,
    and how tight the numerical claim is."""
    delegate = _Delegate(
        _contract(), target="nvidia", op=C.OP_MATMUL,
        variants={"float16": _contract(callee="k_f16", tolerance=1e-4)})
    region = fusion_core.MatmulRegion(dtype="float16")
    assert delegate.contract_for(region).callee == "k_f16"
    assert delegate.accuracy_budget(region)[0] == 1e-4


def test_an_undeclared_dtype_falls_back_to_a_real_contract():
    """The fallback is the representative -- itself a declared, validated
    contract -- so an unlisted dtype still carries a bound rather than
    defaulting to none (Decision #21a: a semantic key never defaults)."""
    delegate = _Delegate(
        _contract(tolerance=7e-3), target="nvidia", op=C.OP_MATMUL,
        variants={"float16": _contract(callee="k_f16")})
    region = fusion_core.MatmulRegion(dtype="float64")
    assert delegate.accuracy_budget(region) == (7e-3, None)


# ── the arbiter consults the per-region budget ───────────────────────────────

def test_the_oracle_adapter_uses_the_region_specific_budget():
    """`_as_runner` used to snapshot the candidate's class-level `accuracy_atol`
    with no region in hand, so a multi-dtype candidate was gated on one dtype's
    number for all of them."""
    delegate = _Delegate(
        _contract(tolerance=1e-1), target="nvidia", op=C.OP_MATMUL,
        variants={"float16": _contract(callee="k_f16", tolerance=1e-6)})
    adapter = C._as_runner(delegate, fusion_core.MatmulRegion(dtype="float16"))
    assert adapter.accuracy_atol == 1e-6


def test_a_plain_candidate_still_answers_with_its_class_attributes():
    """The hook must not disturb backends that register one candidate per
    dtype -- which is most of them."""
    for c in C.candidates_for("nvidia", C.OP_MATMUL):
        region = fusion_core.MatmulRegion(dtype="float16")
        if contract_for_candidate(c) is None:
            assert c.accuracy_budget(region) == (c.accuracy_atol, c.accuracy_rtol)
