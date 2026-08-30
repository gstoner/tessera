"""The Python and C++ halves of the delegation contract must agree.

Two enforcers of one contract is the shape that produced the defect CLAUDE.md
names for Apple — "the Python synthesizer and the C++ MLIR pipeline are two
disconnected compilers". It is only a bridge, rather than a second seam, if
every input one rejects the other rejects too.

`REJECTION_CASES` is therefore written once and used twice: here against the
Python validator, and by the lit fixture
`nvidia_delegate_contract_invalid.mlir` against the ODS verifier. A case added
to one side without the other is the drift this file exists to catch.
"""

from __future__ import annotations

import pytest

from tessera.compiler.emit.candidate import Tier
from tessera.compiler.emit.delegate_contract import (
    DelegateContract,
    DelegateContractError,
    contract_for_candidate,
)


def _call(**overrides):
    base = dict(
        callee="tessera_nvidia_flash",
        arch="sm_120",
        binding="cuda_kernel",
        provenance="handwritten_kernel",
        accuracy="reference_exact",
        determinism="deterministic",
        covers="whole_region",
    )
    base.update(overrides)
    return base


def _ptx(**overrides):
    base = dict(
        ptx="mul.f32 $0, $1, $1;",
        constraints="=f,f",
        arch="sm_120",
        accuracy="reference_exact",
        determinism="deterministic",
        covers="whole_region",
    )
    base.update(overrides)
    return base


#: (id, kwargs, expected-message-fragment). Each has a matching case in
#: nvidia_delegate_contract_invalid.mlir.
REJECTION_CASES = [
    ("empty_callee", _call(callee=""), "non-empty `callee`"),
    ("bounded_without_a_bound",
     _call(accuracy="tolerance_bounded"), "requires `tolerance` and/or"),
    ("exact_carrying_a_tolerance",
     _call(tolerance=1e-3), "must not carry a tolerance"),
    ("non_positive_tolerance",
     _call(accuracy="tolerance_bounded", tolerance=0.0),
     "finite and greater than zero"),
    ("unknown_binding", _call(binding="carrier_pigeon"), "`binding` must be one of"),
    ("unknown_provenance", _call(provenance="rumour"), "`provenance` must be one of"),
    ("empty_constraints", _ptx(constraints=""), "non-empty `constraints`"),
    ("empty_ptx", _ptx(ptx=""), "non-empty `ptx`"),
    ("empty_arch", _call(arch=""), "non-empty `arch`"),
]


@pytest.mark.parametrize(
    "kwargs,fragment",
    [(c[1], c[2]) for c in REJECTION_CASES],
    ids=[c[0] for c in REJECTION_CASES],
)
def test_python_rejects_what_the_ods_verifier_rejects(kwargs, fragment):
    with pytest.raises(DelegateContractError, match=fragment):
        DelegateContract(**kwargs)


def test_the_lit_fixture_covers_the_same_cases():
    """Drift gate across the seam.

    The C++ verifier is exercised by a lit fixture this test cannot run, so it
    checks the cheap invariant instead: every rejection reason asserted here
    appears in that fixture. A case added on the Python side alone would make
    the two enforcers disagree silently, which is the failure mode the whole
    contract exists to prevent.
    """
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    fixture = (
        root / "src/compiler/codegen/tessera_gpu_backend_NVIDIA/test/nvidia"
        / "nvidia_delegate_contract_invalid.mlir"
    ).read_text(encoding="utf-8")

    # Reasons the C++ verifier emits by hand.
    verifier_authored = {
        "non-empty `callee`",
        "requires `tolerance` and/or",
        "must not carry a tolerance",
        "finite and greater than zero",
        "non-empty `constraints`",
        "non-empty `ptx`",
        "non-empty `arch`",
    }
    # Reasons the ODS *constraint* machinery emits, for the string-enum
    # attributes. These reject at parse time rather than in verify(), so they
    # need their own fixture cases or the enum would be untested on that side.
    ods_constrained = {
        "attribute 'binding' failed to satisfy constraint",
        "attribute 'provenance' failed to satisfy constraint",
    }
    missing = sorted(r for r in verifier_authored | ods_constrained if r not in fixture)
    assert not missing, f"rejection reasons not covered by the lit fixture: {missing}"

    # And every Python rejection case must have a fixture counterpart, so the
    # two enforcers cannot drift apart by one side growing a case alone.
    assert len(REJECTION_CASES) == fixture.count("func.func @"), (
        f"{len(REJECTION_CASES)} Python rejection cases but "
        f"{fixture.count('func.func @')} lit cases — the enforcers have drifted"
    )


def test_accepts_both_pathways():
    call = DelegateContract(**_call())
    assert not call.is_inline_asm
    assert call.op_name() == "tessera_nvidia.kernel_call"

    ptx = DelegateContract(**_ptx(accuracy="tolerance_bounded", tolerance=1e-6))
    assert ptx.is_inline_asm
    assert ptx.op_name() == "tessera_nvidia.inline_ptx"


def test_a_delegate_cannot_be_both_pathways():
    """The two ops are exclusive; a hybrid would need a verifier that decides
    which half of its own attributes to trust."""
    with pytest.raises(DelegateContractError, match="not both"):
        DelegateContract(
            arch="sm_120", accuracy="reference_exact",
            determinism="deterministic", covers="whole_region",
            callee="x", binding="c_abi", provenance="vendor_library",
            ptx="mul.f32 $0, $1, $1;", constraints="=f,f",
        )


def test_a_delegate_must_be_one_of_the_pathways():
    with pytest.raises(DelegateContractError, match="must name a `callee` or carry"):
        DelegateContract(arch="sm_120", accuracy="reference_exact",
                         determinism="deterministic", covers="whole_region")


def test_both_provenances_are_tier_three():
    """Origin differs; what the arbiter must do with them does not.

    Neither came from the compiler, so both are scored *against* compiled
    output rather than trusted above it.
    """
    for provenance in ("vendor_library", "handwritten_kernel"):
        contract = DelegateContract(**_call(provenance=provenance))
        assert contract.arbiter_tier() is Tier.HAND_TUNED


def test_accuracy_budget_is_derived_from_the_contract_not_hand_set():
    """The budget the oracle holds a delegate to comes from what it declared."""
    bounded = DelegateContract(**_call(accuracy="tolerance_bounded", tolerance=2.5e-4))
    assert bounded.arbiter_accuracy_atol() == 2.5e-4

    exact = DelegateContract(**_call())
    # None means "the oracle's default budget" -- the same standard compiled
    # output is held to. Deliberately not 0.0, which would reject a correct
    # candidate, since an exact claim is "no worse than the reference" rather
    # than "bit-identical".
    assert exact.arbiter_accuracy_atol() is None


def test_rendered_op_carries_every_arbiter_input():
    contract = DelegateContract(**_call(accuracy="tolerance_bounded", tolerance=1e-6))
    text = contract.render_op("%a", "(f32) -> f32")
    assert text.startswith("tessera_nvidia.kernel_call %a")
    for fragment in ('callee = "tessera_nvidia_flash"', 'binding = "cuda_kernel"',
                     'provenance = "handwritten_kernel"', 'arch = "sm_120"',
                     'accuracy = "tolerance_bounded"', "tolerance = 1.000000e-06 : f64"):
        assert fragment in text, fragment


def test_inline_ptx_side_effects_are_explicit():
    """PTX touching memory or barriers must not be reordered away; absence of
    evidence is not purity."""
    quiet = DelegateContract(**_ptx())
    assert "has_side_effects" not in quiet.render_attributes()
    loud = DelegateContract(**_ptx(has_side_effects=True))
    assert "has_side_effects" in loud.render_attributes()


def test_a_non_delegate_candidate_has_no_contract():
    """Inventing a contract for compiler-generated work would erase the exact
    distinction `provenance` exists to record."""
    class _Plain:
        pass

    assert contract_for_candidate(_Plain()) is None


def test_delegated_candidate_derives_tier_and_budget_from_the_contract():
    """A delegate cannot claim a budget in Python it did not declare in IR.

    Hand-setting these at the registration site is how the Python and C++
    halves of a compiler drift apart, so they are derived rather than passed.
    """
    from tessera.compiler.emit.delegate_contract import DelegatedCandidate

    class _Lib(DelegatedCandidate):
        def run(self, region, *inputs, **kwargs):  # pragma: no cover - unused
            raise NotImplementedError

    contract = DelegateContract(
        **_call(provenance="vendor_library", accuracy="tolerance_bounded",
                tolerance=5e-4))
    candidate = _Lib(contract, target="nvidia", op="matmul")

    assert candidate.tier is Tier.HAND_TUNED
    assert candidate.accuracy_atol == 5e-4
    assert candidate.name == "tessera_nvidia_flash:sm_120"
    assert contract_for_candidate(candidate) is contract
    assert "provenance = \"vendor_library\"" in candidate.render_target_ir()


def test_determinism_must_be_declared():
    """Tessera guarantees @jit(deterministic=True); a delegate can break it.

    A split-K kernel accumulating with atomics is not reproducible run to run.
    Without a declared determinism the arbiter could select one inside a
    deterministic region -- a guarantee defeated through a path nobody checked,
    which is the Decision #5 scar exactly.
    """
    import dataclasses

    nd = DelegateContract(**_call(determinism="nondeterministic"))
    assert nd.is_deterministic() is False
    assert 'determinism = "nondeterministic"' in nd.render_attributes()
    assert DelegateContract(**_call()).is_deterministic() is True

    with pytest.raises(DelegateContractError, match="`determinism` must be one of"):
        DelegateContract(**_call(determinism="probably"))
    # and it cannot simply be omitted
    with pytest.raises(TypeError):
        kwargs = _call()
        del kwargs["determinism"]
        DelegateContract(**kwargs)
    assert dataclasses.is_dataclass(nd)


def test_a_relative_only_claim_is_expressible():
    """An absolute bound alone is meaningless without the result's magnitude.

    1e-6 is vacuous on values of order 1e6 and unsatisfiable on 1e-9, so a
    delegate whose real claim is relative must be able to state it rather than
    overclaim in absolute terms. The arbiter's Candidate already carried both
    atol and rtol; the contract initially carried only atol.
    """
    rel = DelegateContract(**_call(accuracy="tolerance_bounded", tolerance_rel=1e-3))
    assert rel.arbiter_accuracy_rtol() == 1e-3
    assert rel.arbiter_accuracy_atol() is None

    mixed = DelegateContract(
        **_call(accuracy="tolerance_bounded", tolerance=1e-6, tolerance_rel=1e-3))
    assert (mixed.arbiter_accuracy_atol(), mixed.arbiter_accuracy_rtol()) == (1e-6, 1e-3)

    with pytest.raises(DelegateContractError, match="must not carry a tolerance"):
        DelegateContract(**_call(tolerance_rel=1e-3))


def test_delegated_candidate_carries_both_budgets():
    from tessera.compiler.emit.delegate_contract import DelegatedCandidate

    class _Lib(DelegatedCandidate):
        def run(self, region, *inputs, **kwargs):  # pragma: no cover
            raise NotImplementedError

    c = _Lib(DelegateContract(**_call(accuracy="tolerance_bounded",
                                      tolerance=1e-5, tolerance_rel=1e-3)),
             target="nvidia", op="matmul")
    assert (c.accuracy_atol, c.accuracy_rtol) == (1e-5, 1e-3)


class _FakeRegion:
    """Minimal stand-in carrying the fields that make a region more than a root."""

    def __init__(self, **fields):
        self.epilogue = fields.get("epilogue", ())
        self.reduction = fields.get("reduction")
        self.prologue = fields.get("prologue", ())
        self.residual = fields.get("residual", False)


def _delegate(covers, op="matmul"):
    from tessera.compiler.emit.delegate_contract import DelegatedCandidate

    class _Lib(DelegatedCandidate):
        def run(self, region, *inputs, **kwargs):  # pragma: no cover
            raise NotImplementedError

    return _Lib(DelegateContract(**_call(covers=covers)), target="nvidia", op=op)


def test_a_root_only_delegate_declines_a_fused_region():
    """The fusion-foreclosure fix, stated as the scenario it prevents.

    `arbitrate()` picks by tier by default, and HAND_TUNED is the highest —
    so a delegate wins outright before anything is measured. On the measured
    path it wins because the latency excludes the work it displaced. Either
    way a bare-GEMM delegate would beat a fused compiled kernel on exactly
    the regions where fusion is the win.
    """
    root_only = _delegate("root_only")
    assert root_only.applies_to(_FakeRegion()) is True
    for structure in (
        {"epilogue": ("bias", "relu")},
        {"reduction": "softmax"},
        {"prologue": ("gelu",)},
        {"residual": True},
    ):
        assert root_only.applies_to(_FakeRegion(**structure)) is False, structure


def test_a_whole_region_delegate_still_competes():
    """Declining must not disarm legitimate fused hand-tuned kernels.

    Decision #28's governing rule is that shared infra never caps the leads; a
    hand-tuned fused attention kernel has to stay a first-class candidate.
    """
    whole = _delegate("whole_region")
    assert whole.applies_to(_FakeRegion(epilogue=("bias",), reduction="softmax")) is True


def test_coverage_must_be_declared_not_guessed():
    """An external kernel cannot be introspected; guessing is the bug."""
    with pytest.raises(DelegateContractError, match="`covers` must be one of"):
        DelegateContract(**_call(covers="probably_all_of_it"))


def test_declining_beats_penalising():
    """Recorded as a design note, asserted so the mechanism cannot drift back.

    A cost penalty for foregone fusion is a guess at DRAM traffic. "This
    candidate does not serve this region" is a fact the delegate declared, and
    it removes the candidate from the comparison rather than hoping a fudge
    factor outweighs a tier bonus.
    """
    root_only = _delegate("root_only")
    fused = _FakeRegion(epilogue=("bias",))
    assert root_only.applies_to(fused) is False
    assert root_only.tier is Tier.HAND_TUNED  # would otherwise win on tier alone
