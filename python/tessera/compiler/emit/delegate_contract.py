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
from typing import Any

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

#: Legal `determinism` values. Tessera guarantees `@jit(deterministic=True)`,
#: and a delegate using split-K with atomic accumulation is not reproducible
#: run to run. Without this the arbiter could silently select such a delegate
#: inside a deterministic region -- a guarantee defeated through a path nobody
#: checked, which is the Decision #5 scar exactly.
DETERMINISMS: tuple[str, ...] = ("deterministic", "nondeterministic")

#: Legal `covers` values — how much of a fusable region the delegate implements.
#:
#: This exists because the arbiter compares candidates *for one region*, and a
#: delegate that implements only the region's root is not offering the same
#: thing more cheaply — it is offering a different plan: the delegate, plus a
#: separate epilogue kernel, plus the DRAM round-trip between them. Treating
#: the two as peers is a category error, and it biases selection toward
#: delegates on exactly the graphs where fusion is the win.
#:
#: Declared rather than inferred: an external kernel cannot be introspected,
#: and guessing its coverage is how a partial candidate wins a whole-region
#: comparison.
COVERS: tuple[str, ...] = ("root_only", "whole_region")


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
    determinism: str
    covers: str
    tolerance: float | None = None
    tolerance_rel: float | None = None
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
        if self.covers not in COVERS:
            raise DelegateContractError(
                f"`covers` must be one of {COVERS}, got {self.covers!r}"
            )
        if self.determinism not in DETERMINISMS:
            raise DelegateContractError(
                f"`determinism` must be one of {DETERMINISMS}, "
                f"got {self.determinism!r}"
            )
        # The budget half of "fastest in-budget candidate". A semantic key:
        # never defaulted, and never self-contradictory (Decision #21a).
        for name in ("tolerance", "tolerance_rel"):
            bound = getattr(self, name)
            if bound is not None and (
                not (bound > 0.0) or bound == float("inf")
            ):
                raise DelegateContractError(
                    f"`{name}` must be finite and greater than zero"
                )
        if self.accuracy == "tolerance_bounded":
            # Absolute OR relative satisfies the claim. An absolute bound alone
            # is meaningless without the result's magnitude, so a delegate whose
            # real claim is relative must be able to say so rather than
            # overclaim in absolute terms.
            if self.tolerance is None and self.tolerance_rel is None:
                raise DelegateContractError(
                    "accuracy=tolerance_bounded requires `tolerance` and/or "
                    "`tolerance_rel`; a bounded numerical claim with no stated "
                    "bound is not a claim the arbiter can budget against"
                )
        elif self.tolerance is not None or self.tolerance_rel is not None:
            raise DelegateContractError(
                "accuracy=reference_exact must not carry a tolerance; an "
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

    def arbiter_accuracy_rtol(self) -> float | None:
        """Relative budget, if the delegate stated one."""
        return self.tolerance_rel

    def serves_whole_region(self) -> bool:
        """Whether this delegate implements a fused region, not just its root."""
        return self.covers == "whole_region"

    def is_deterministic(self) -> bool:
        """Whether this delegate may be selected inside a deterministic region."""
        return self.determinism == "deterministic"

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
        parts.append(f'determinism = "{self.determinism}"')
        parts.append(f'covers = "{self.covers}"')
        if self.tolerance is not None:
            parts.append(f"tolerance = {self.tolerance:.6e} : f64")
        if self.tolerance_rel is not None:
            parts.append(f"tolerance_rel = {self.tolerance_rel:.6e} : f64")
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

    #: Contract fields that describe the delegate itself rather than one dtype
    #: route. Every member of a `variants` family must agree on these, or the
    #: representative contract misdescribes the family.
    _FAMILY_INVARIANT = ("arch", "accuracy", "determinism", "covers",
                         "binding", "provenance")

    def __init__(self, contract: DelegateContract, *, target: str, op: str,
                 name: str | None = None,
                 variants: "dict[str, DelegateContract] | None" = None) -> None:
        self.delegate_contract = contract
        # A delegate that binds a DIFFERENT symbol per dtype has more than one
        # identity, and `callee` is identity. The shipped NVIDIA GEMM is the
        # first real case: one candidate reaching
        # `tessera_nvidia_mma_gemm_f16` or `..._bf16` by dtype. Declaring one
        # of those callees and sometimes calling the other is exactly the
        # Python-vs-IR drift this contract exists to stop, so the family is
        # declared instead.
        #
        # Callee identity is the WHOLE justification. A per-dtype *tolerance*
        # looked like a second one -- bf16 carries 8 significand bits to f16's
        # 11 -- and measurement on sm_120 refuted it: the two agree to within
        # 25% at every K from 32 to 8192. The reason is that the oracle's
        # reference rounds its operands to the storage dtype first, so input
        # rounding cancels on both sides and the residual is f32
        # accumulation-order error, which does not depend on the storage
        # dtype. The hook is still per-dtype because a delegate family whose
        # members really do differ numerically can say so; this one does not.
        self.contract_variants = dict(variants or {})
        for dtype, variant in self.contract_variants.items():
            differing = [f for f in self._FAMILY_INVARIANT
                         if getattr(variant, f) != getattr(contract, f)]
            if differing:
                raise DelegateContractError(
                    f"delegate variant {dtype!r} disagrees with the family on "
                    f"{', '.join(differing)}; those fields describe the "
                    "delegate, not one dtype route, so a representative "
                    "contract carrying different values would misdescribe it"
                )
        # `name` is a dispatch/cache key, NOT a claim -- so unlike tier and
        # budget it is the registrant's to choose. Deriving it from `callee`
        # is a good default and a bad requirement: the autotune corpus and the
        # E3 `force` escape hatch key on this string, so binding it to a C
        # symbol means renaming that symbol silently invalidates every
        # persisted verdict and breaks `force` with no error. First use found
        # this -- the shipped NVIDIA GEMM already had a stable name predating
        # its contract.
        self.name = name or contract.identity()
        self.target = target
        self.op = op
        # Derived, never assigned by the subclass. These ARE claims: a delegate
        # must not be able to assert a tier or a budget in Python that it did
        # not declare to the verifier.
        self.tier = contract.arbiter_tier()
        self.accuracy_atol = contract.arbiter_accuracy_atol()
        self.accuracy_rtol = contract.arbiter_accuracy_rtol()

    #: Fields whose presence means a `FusedRegion` is more than its root.
    _FUSED_STRUCTURE = ("epilogue", "reduction", "prologue", "residual")

    #: Region classes that are fused **by construction**, whatever their fields
    #: say. `AttentionRegion` is softmax(QKᵀ)·V — two matmuls and a softmax —
    #: and carries none of `_FUSED_STRUCTURE`, so probing those names alone
    #: reports it as a bare root and re-admits the very bias this method exists
    #: to remove (review finding on PR #650). Declared rather than derived
    #: because "how many operations is this region" is not a question the
    #: dataclasses answer; verified against `fusion_core` so a rename fails
    #: loudly instead of silently widening what a partial delegate may serve.
    _INTRINSICALLY_FUSED = (
        "AttentionRegion",
        "GatedMatmulRegion",
        "NormChainRegion",
        "PointwiseReduceRegion",
    )

    #: Regions carrying an explicit operation list are fused when it holds more
    #: than one op — `PointwiseGraphRegion` is a chain, and a single-op chain is
    #: the only shape a root-only delegate can serve whole.
    _OP_LIST_FIELD = "ops"

    def applies_to(self, region: Any) -> bool:
        """Decline a region this delegate implements only part of.

        **This is what keeps the arbiter from being biased toward delegates**,
        and it is a structural fact rather than a cost estimate.

        `arbitrate()` compares candidates for one region and picks by tier
        (hand-tuned wins by default) or by measured latency of the candidate in
        isolation. A `root_only` delegate — a bare GEMM, say — faced with a
        matmul+epilogue region is not a cheaper way to do the same work. It is
        a *different plan*: the delegate, plus a separate epilogue kernel, plus
        the DRAM round-trip between them. Under tier priority it would win
        outright; under measured latency it would win because the measurement
        excludes the work it displaced. Both paths would systematically prefer
        delegates on exactly the graphs where fusion is the win.

        Declining is the honest answer rather than applying a penalty: a
        penalty is a guess at the foregone fusion, whereas "this candidate does
        not serve this region" is a fact the delegate declared. If the
        delegate-plus-epilogue plan really is faster, that belongs in a
        comparison of *plans*, not smuggled in as a peer candidate here.

        **Fails closed on an unrecognised region.** The first version of this
        method probed four `FusedRegion` field names and returned `True` when
        none were set — so every region class that does not have those fields,
        including `AttentionRegion`, was reported as a bare root and admitted a
        partial delegate anyway. An unknown region shape is now treated as
        fused: being wrong in that direction costs a candidate its slot, while
        being wrong the other way silently restores the bias.
        """
        if self.delegate_contract.serves_whole_region():
            return True
        return _region_is_a_single_operation(
            region,
            fused_structure=self._FUSED_STRUCTURE,
            intrinsically_fused=self._INTRINSICALLY_FUSED,
            op_list_field=self._OP_LIST_FIELD,
        )

    def contract_for(self, region: Any) -> DelegateContract:
        """The contract governing this delegate for `region`.

        Falls back to the representative when the region names no dtype or the
        family declares no variant for it -- the representative is a real
        declared contract, so the fallback still carries a bound rather than
        defaulting to none (Decision #21a).
        """
        dtype = getattr(region, "dtype", None)
        if isinstance(dtype, str):
            variant = self.contract_variants.get(dtype)
            if variant is not None:
                return variant
        return self.delegate_contract

    def accuracy_budget(self, region: Any) -> "tuple[float | None, float | None]":
        """`(atol, rtol)` for `region`, from that region's declared contract."""
        c = self.contract_for(region)
        return c.arbiter_accuracy_atol(), c.arbiter_accuracy_rtol()

    def render_target_ir(self, operands: str = "",
                         signature: str = "() -> ()") -> str:
        """The Target IR op declaring this candidate's delegation."""
        return self.delegate_contract.render_op(operands, signature)


#: Region classes a root-only delegate may serve, when otherwise unstructured.
#: Anything not named here and not matching a known single-op shape is treated
#: as fused — see `verify_region_classes`.
_SINGLE_OP_REGIONS: tuple[str, ...] = ("MatmulRegion", "FusedRegion",
                                       "PointwiseGraphRegion")


def _region_is_a_single_operation(
    region: Any, *, fused_structure: tuple[str, ...],
    intrinsically_fused: tuple[str, ...], op_list_field: str,
) -> bool:
    """Whether `region` is one operation, so a root-only delegate serves it whole.

    Fails closed: an unrecognised region class is fused. Three shapes are
    recognised as possibly-single, and each is checked structurally rather than
    trusted by name.
    """
    name = type(region).__name__
    if name in intrinsically_fused:
        return False
    if name not in _SINGLE_OP_REGIONS:
        return False  # unknown shape — assume fused
    ops = getattr(region, op_list_field, None)
    if ops is not None:
        return len(ops) <= 1
    return not any(getattr(region, field, None) for field in fused_structure)


def verify_region_classes() -> None:
    """Fail loudly if a named region class no longer exists.

    Both tables above are declared, not derived. A renamed or deleted region
    would silently drop out of `_INTRINSICALLY_FUSED` and be re-admitted as a
    bare root — restoring the bias with no test failing, because the wrong
    answer is a `True` rather than an exception.
    """
    from tessera.compiler import fusion_core

    missing = [
        n for n in (*_SINGLE_OP_REGIONS,
                    *DelegatedCandidate._INTRINSICALLY_FUSED)
        if not hasattr(fusion_core, n)
    ]
    if missing:
        raise RuntimeError(
            "delegate_contract: region classes no longer in fusion_core: "
            + ", ".join(sorted(missing))
            + ". Update the tables rather than letting a partial delegate be "
            "admitted to a fused region."
        )


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
    "DETERMINISMS",
    "COVERS",
    "DelegateContractError",
    "DelegatedCandidate",
    "contract_for_candidate",
]
