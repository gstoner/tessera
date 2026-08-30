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
    )
    base.update(overrides)
    return base


def _ptx(**overrides):
    base = dict(
        ptx="mul.f32 $0, $1, $1;",
        constraints="=f,f",
        arch="sm_120",
        accuracy="reference_exact",
    )
    base.update(overrides)
    return base


#: (id, kwargs, expected-message-fragment). Each has a matching case in
#: nvidia_delegate_contract_invalid.mlir.
REJECTION_CASES = [
    ("empty_callee", _call(callee=""), "non-empty `callee`"),
    ("bounded_without_a_bound",
     _call(accuracy="tolerance_bounded"), "requires a `tolerance`"),
    ("exact_carrying_a_tolerance",
     _call(tolerance=1e-3), "must not carry a `tolerance`"),
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
        "requires a `tolerance`",
        "must not carry a `tolerance`",
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
            callee="x", binding="c_abi", provenance="vendor_library",
            ptx="mul.f32 $0, $1, $1;", constraints="=f,f",
        )


def test_a_delegate_must_be_one_of_the_pathways():
    with pytest.raises(DelegateContractError, match="must name a `callee` or carry"):
        DelegateContract(arch="sm_120", accuracy="reference_exact")


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
