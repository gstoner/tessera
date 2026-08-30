"""One delegation contract, enforced on both sides of the compiler seam.

`tessera_nvidia.kernel_call` and `tessera_nvidia.inline_ptx` declare a fast
path *out* of the compiler — a vendor library entry, a hand-tuned kernel, or
inline PTX — in a form Decision #28's arbiter can score. Their ODS verifier
lives in C++ (`TesseraNVIDIADialect.cpp`). This module is the Python half.

Why both halves exist, and why that is not a duplication
--------------------------------------------------------
CLAUDE.md names the defect this avoids: *"the Python synthesizer and the C++
MLIR pipeline are two disconnected compilers."* A delegation boundary would
reproduce it exactly if Python emitted the ops from one set of rules while the
verifier enforced another — the emitter would produce IR the compiler rejects,
or worse, IR it accepts for the wrong reason.

So this is deliberately **not** a second contract. It is the same contract
expressed where the arbiter can use it, with `test_delegate_contract.py`
asserting the two enforcers agree case for case — every input the C++ verifier
rejects is rejected here, with the same reason. That differential is the point;
without it this file would be the seam rather than a bridge across it.

What the arbiter gets
---------------------
The arbiter already models both facts the contract declares — `Candidate.tier`
is Decision #28's provenance tier, and `accuracy_atol` is the budget half of
"fastest *in-budget* candidate". A declared delegate therefore does not need a
bespoke scoring path: it needs its tier and budget *derived from the contract*
rather than hand-set at the registration site, so a delegate cannot claim in
Python a budget it did not declare in IR.
"""

from __future__ import annotations

from dataclasses import dataclass

from tessera.compiler.emit.candidate import Candidate, Tier

#: Legal `binding` values — how the delegate is reached. Mirrors
#: NVIDIA_DelegateBindingAttr.
BINDINGS: tuple[str, ...] = ("cuda_kernel", "c_abi")

#: Legal `provenance` values — what the delegate is. Mirrors
#: NVIDIA_DelegateProvenanceAttr.
PROVENANCES: tuple[str, ...] = ("vendor_library", "handwritten_kernel")

#: Legal `accuracy` values — the numerical claim. Mirrors
#: NVIDIA_DelegateAccuracyAttr.
ACCURACIES: tuple[str, ...] = ("reference_exact", "tolerance_bounded")


class DelegateContractError(ValueError):
    """A delegate that the Target IR verifier would reject.

    Raised in Python so a bad delegate fails at construction rather than
    producing IR that dies later in `tessera-opt`, where the diagnostic is far
    from the emitter that caused it.
    """


@dataclass(frozen=True)
class DelegateContract:
    """A fast path out of the compiler, declared so the arbiter can score it.

    Exactly one of `callee` (a named CUDA kernel / host C-ABI symbol) or `ptx`
    (inline PTX text) is set — those are the two pathways, and they are
    separate ops in the dialect for the same reason they are exclusive here:
    an unresolved symbol and an empty asm body are different failures.
    """

    arch: str
    accuracy: str
    tolerance: float | None = None
    # kernel_call pathway
    callee: str | None = None
    binding: str | None = None
    provenance: str | None = None
    # inline_ptx pathway
    ptx: str | None = None
    constraints: str | None = None
    has_side_effects: bool = False

    def __post_init__(self) -> None:
        self.validate()

    # -- validation: must match the C++ verifier case for case ---------------

    def validate(self) -> None:
        """Reject exactly what `TesseraNVIDIADialect.cpp` rejects."""
        if not self.arch:
            raise DelegateContractError("requires a non-empty `arch`")

        is_call = self.callee is not None or self.binding is not None \
            or self.provenance is not None
        is_ptx = self.ptx is not None or self.constraints is not None
        if is_call and is_ptx:
            raise DelegateContractError(
                "a delegate is either a named call or inline ptx, not both; "
                "they are separate ops in the dialect"
            )
        if not is_call and not is_ptx:
            raise DelegateContractError(
                "a delegate must name a `callee` or carry `ptx`"
            )

        if self.accuracy not in ACCURACIES:
            raise DelegateContractError(
                f"`accuracy` must be one of {ACCURACIES}, got {self.accuracy!r}"
            )
        # The budget half of "fastest in-budget candidate". A semantic key:
        # never defaulted, and never self-contradictory (Decision #21a).
        if self.accuracy == "tolerance_bounded":
            if self.tolerance is None:
                raise DelegateContractError(
                    "accuracy=tolerance_bounded requires a `tolerance`; a "
                    "bounded numerical claim with no stated bound is not a "
                    "claim the arbiter can budget against"
                )
            if not (self.tolerance > 0.0) or self.tolerance == float("inf"):
                raise DelegateContractError(
                    "`tolerance` must be finite and greater than zero"
                )
        elif self.tolerance is not None:
            raise DelegateContractError(
                "accuracy=reference_exact must not carry a `tolerance`; an "
                "exact claim with a tolerance is two contradictory claims"
            )

        if is_call:
            if not self.callee:
                raise DelegateContractError(
                    "requires a non-empty `callee`; a delegation with no named "
                    "target cannot be bound, cached, or re-measured"
                )
            if self.binding not in BINDINGS:
                raise DelegateContractError(
                    f"`binding` must be one of {BINDINGS}, got {self.binding!r}"
                )
            if self.provenance not in PROVENANCES:
                raise DelegateContractError(
                    f"`provenance` must be one of {PROVENANCES}, "
                    f"got {self.provenance!r}"
                )
        else:
            if not self.ptx:
                raise DelegateContractError(
                    "requires non-empty `ptx`; an empty inline-asm body is a "
                    "silently successful no-op rather than an error"
                )
            if not self.constraints:
                raise DelegateContractError(
                    "requires a non-empty `constraints` string; inline asm "
                    "whose operand constraints are unstated cannot be bound "
                    "correctly"
                )

    # -- what the arbiter reads ----------------------------------------------

    @property
    def is_inline_asm(self) -> bool:
        return self.ptx is not None

    def arbiter_tier(self) -> Tier:
        """Decision #28 tier. Both delegate kinds are Tier 3.

        A vendor library and an in-tree hand-tuned kernel differ in origin but
        not in what the arbiter must do with them: neither came from the
        compiler, so both are scored *against* compiled output rather than
        trusted above it. `provenance` is still carried because it is the fact
        that distinguishes them in a dispatch log and in a report.
        """
        return Tier.HAND_TUNED

    def arbiter_accuracy_atol(self) -> float | None:
        """Absolute budget the F4 oracle must hold this delegate to.

        `reference_exact` returns ``None``, which the arbiter reads as "the
        oracle's default budget" — the same standard compiled output is held
        to. It deliberately does not return 0.0: an exact *claim* means "no
        worse than the reference", not "bit-identical in floating point", and
        a zero budget would reject every candidate including a correct one.
        """
        return self.tolerance if self.accuracy == "tolerance_bounded" else None

    def identity(self) -> str:
        """Stable name for caching, dispatch logs, and the `force` escape hatch."""
        return f"inline_ptx:{self.arch}" if self.is_inline_asm \
            else f"{self.callee}:{self.arch}"

    # -- MLIR rendering -------------------------------------------------------

    def render_attributes(self) -> str:
        """The op's attribute dictionary, in the dialect's spelling."""
        parts: list[str] = []
        if self.is_inline_asm:
            parts.append(f'ptx = "{self.ptx}"')
            parts.append(f'constraints = "{self.constraints}"')
        else:
            parts.append(f'callee = "{self.callee}"')
            parts.append(f'binding = "{self.binding}"')
            parts.append(f'provenance = "{self.provenance}"')
        parts.append(f'arch = "{self.arch}"')
        parts.append(f'accuracy = "{self.accuracy}"')
        if self.tolerance is not None:
            parts.append(f"tolerance = {self.tolerance:.6e} : f64")
        if self.is_inline_asm and self.has_side_effects:
            parts.append("has_side_effects")
        return "{" + ", ".join(parts) + "}"

    def op_name(self) -> str:
        return "tessera_nvidia.inline_ptx" if self.is_inline_asm \
            else "tessera_nvidia.kernel_call"

    def render_op(self, operands: str = "", signature: str = "() -> ()") -> str:
        """One Target IR operation declaring this delegate."""
        lead = f"{self.op_name()} {operands}".rstrip()
        return f"{lead} {self.render_attributes()} : {signature}"


class DelegatedCandidate(Candidate):
    """An arbiter candidate whose tier and budget come from its IR contract.

    This is the integration point, and the reason it is a base class rather
    than a helper: a delegate author cannot hand-set `tier` or
    `accuracy_atol`. Both are derived from the `DelegateContract` the delegate
    declares in Target IR, so a candidate cannot claim a budget in Python that
    it did not declare to the verifier — which is precisely how the two halves
    of a compiler drift apart.

    Subclasses supply `run()` (binding the symbol or emitting the asm) and,
    where the delegate is only present on real silicon, `available()`. Nothing
    else about arbitration changes: a delegate is enumerated, F4-gated, and
    selected by the same `arbitrate()` path as compiled candidates, which is
    the point — Decision #28 scores a hand-tuned kernel *against* compiled
    output rather than trusting it above.
    """

    def __init__(self, contract: DelegateContract, *, target: str, op: str) -> None:
        self.delegate_contract = contract
        self.name = contract.identity()
        self.target = target
        self.op = op
        # Derived, never assigned by the subclass.
        self.tier = contract.arbiter_tier()
        self.accuracy_atol = contract.arbiter_accuracy_atol()

    def render_target_ir(self, operands: str = "",
                         signature: str = "() -> ()") -> str:
        """The Target IR op declaring this candidate's delegation."""
        return self.delegate_contract.render_op(operands, signature)


def contract_for_candidate(candidate) -> DelegateContract | None:
    """The contract a registered candidate declares, or ``None``.

    A candidate that is not a declared delegate returns ``None`` rather than a
    fabricated contract — the arbiter must be able to tell "compiler-generated"
    from "delegated", and inventing a contract for the former would erase
    exactly the distinction `provenance` exists to record.
    """
    return getattr(candidate, "delegate_contract", None)


__all__ = [
    "ACCURACIES",
    "BINDINGS",
    "PROVENANCES",
    "DelegateContract",
    "DelegateContractError",
    "DelegatedCandidate",
    "contract_for_candidate",
]
