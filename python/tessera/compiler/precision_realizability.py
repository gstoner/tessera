"""NUMPOL-CARRIER-1 (queue row 3b) / FORGE §W5 — is the fused-epilogue
fp32-accumulator precision benefit actually realizable?

This is the acceptance target queue row 3b names: the verdict must be decided
by the **carried `numeric_policy`** together with the optimizer-state dtype,
not by a special case. It is the strongest single argument for the carrier —
a compiler can answer this and a hand-written kernel cannot, because only the
compiler sees both the accumulator contract and the state dtypes at once.

── The finding it encodes ──

`docs/audit/compiler/FORGE_ASSESSMENT.md` §1.3 isolates gradient rounding over
20 steps against an fp64 reference:

    fp32 master W, fp32 states    standard 2.970e-04   FORGE 3.254e-07   913x
    fp32 master W, bf16 states    standard 8.053e-04   FORGE 7.508e-04     1.1x
    bf16 W, bf16 states           standard 7.371e-03   FORGE 7.380e-03     1.0x  <- the paper's own recipe

bf16 **state** rounding swamps the gradient rounding entirely, so the paper
measures its precision claim inside a recipe that masks it. The benefit is a
function of accumulator x state dtype, and in the memory-saving configurations
the fusion is advertised for, it is not there.

── What this module claims, and what it does not ──

Reproduced here (`tests/unit/test_precision_realizability.py`) with the
gradient stored in bf16 as mixed precision does: **208x / 1.2x / 1.0x**. The
masked rows match the assessment closely (1.2 vs 1.1, 1.0 vs 1.0); the
unmasked row agrees in kind but not magnitude, because how large the benefit
gets depends on the gradient distribution and step count, which a compiler does
not know.

That asymmetry decides the interface. The oracle answers the question it can
answer soundly — **is the benefit masked?** — with a number, and refuses to
put a number on the unmasked case. A diagnostic that said "expect 913x" and
delivered 208x would be worse than one that said "large; measure it".

The model: each rounding site contributes an independent relative error of
about its unit roundoff, and they combine in quadrature. Removing the
gradient's write to storage therefore improves the weight error by

    sqrt(eps_grad^2 + eps_state^2 + eps_master^2) / sqrt(eps_state^2 + eps_master^2)

which is exactly right when a surviving term dominates (the masked case) and
an over-estimate when none does (the unmasked case, where the ratio is set by
dynamics this expression does not model).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

__all__ = [
    "RealizabilityVerdict",
    "fused_epilogue_realizability",
    "significand_bits",
]

#: Significand bits INCLUDING the implicit leading one. Mirrors the C++
#: `numericPolicyMantissaBits` in IRContractLegalityPass — the accumulator's
#: contract is precision, not width, so fp16 (11) and bf16 (8) differ here
#: despite both being 16-bit.
_SIGNIFICAND_BITS = {
    "fp64": 53, "f64": 53,
    "fp32": 24, "f32": 24,
    "tf32": 11,
    "fp16": 11, "f16": 11,
    "bf16": 8,
    "fp8_e4m3": 4, "fp8_e5m2": 3,
    "fp6_e2m3": 4, "fp6_e3m2": 3,
    "fp4_e2m1": 2, "nvfp4": 2,
}

#: Below this ratio the benefit is masked: the surviving rounding sources are
#: within a factor of two of the one being removed, so removing it changes
#: little. Chosen to sit above the measured masked rows (1.2x, 1.0x) and well
#: below the unmasked one (208x) — a gap of two orders, so the threshold is
#: not doing delicate work.
_MASKED_BELOW = 2.0


def significand_bits(dtype: str) -> int:
    """Significand bits for a canonical Tessera dtype name."""
    try:
        return _SIGNIFICAND_BITS[str(dtype)]
    except KeyError as exc:  # semantic key: no silent default (#21a)
        raise ValueError(
            f"precision realizability: unknown dtype {dtype!r}; it has no "
            f"stated significand width, and guessing one would put a number "
            f"on a verdict that has no basis"
        ) from exc


def _unit_roundoff(dtype: str) -> float:
    return 2.0 ** -significand_bits(dtype)


@dataclass(frozen=True)
class RealizabilityVerdict:
    """Whether fusing the optimizer into the matmul epilogue actually buys
    precision, for this policy and these state dtypes."""

    realizable: bool
    #: Expected weight-error improvement. ``None`` when the benefit is real
    #: but its size depends on dynamics this model does not capture — see the
    #: module docstring on why a number is withheld rather than invented.
    expected_improvement: Optional[float]
    #: Which rounding site dominates what SURVIVES the fusion.
    dominant_surviving: str
    explanation: str

    def diagnostic(self) -> str:
        return self.explanation


def fused_epilogue_realizability(
    numeric_policy: Mapping[str, Any] | None,
    *,
    state_dtype: str,
    master_dtype: str,
    gradient_storage: str | None = None,
) -> RealizabilityVerdict:
    """Decide the verdict from the CARRIED policy plus the state dtypes.

    ``gradient_storage`` defaults to the policy's ``storage``: the fusion
    removes the gradient's round-trip through that dtype, so that is the
    rounding it eliminates. ``accum`` is what the gradient is produced in and
    what the fused epilogue hands to the optimizer.
    """
    policy = dict(numeric_policy or {})
    accum = policy.get("accum")
    storage = gradient_storage if gradient_storage is not None else policy.get("storage")
    if accum is None or storage is None:
        raise ValueError(
            "precision realizability needs both numeric_policy.accum and a "
            "gradient storage dtype: the verdict IS the relationship between "
            "them, so an absent one makes the question unanswerable rather "
            "than defaultable (#21a)"
        )

    eps_grad = _unit_roundoff(str(storage))
    eps_accum = _unit_roundoff(str(accum))
    eps_state = _unit_roundoff(state_dtype)
    eps_master = _unit_roundoff(master_dtype)

    # The fused path still rounds the gradient to the ACCUMULATOR; it removes
    # only the extra rounding to storage. If the accumulator is no better than
    # the storage, there is nothing to remove in the first place.
    if eps_accum >= eps_grad:
        return RealizabilityVerdict(
            realizable=False,
            expected_improvement=1.0,
            dominant_surviving="accumulator",
            explanation=(
                f"No benefit: accum={accum!r} is no more precise than the "
                f"gradient storage {storage!r}, so fusing removes no rounding. "
                f"Widen accum before expecting the fusion to buy precision."
            ),
        )

    surviving = (eps_state ** 2 + eps_master ** 2 + eps_accum ** 2) ** 0.5
    with_grad_rounding = (surviving ** 2 + eps_grad ** 2) ** 0.5
    ratio = with_grad_rounding / surviving

    dominant, dom_eps = max(
        (("optimizer state", eps_state), ("master weights", eps_master),
         ("accumulator", eps_accum)),
        key=lambda pair: pair[1],
    )

    if ratio < _MASKED_BELOW:
        return RealizabilityVerdict(
            realizable=False,
            expected_improvement=ratio,
            dominant_surviving=dominant,
            explanation=(
                f"Masked: expected weight-error improvement is only "
                f"~{ratio:.2f}x. Removing the gradient's round-trip through "
                f"{storage!r} (unit roundoff {eps_grad:.2e}) does not help "
                f"while {dominant} at {dom_eps:.2e} still rounds every step. "
                f"The fusion may still be worth it for bandwidth; it is not "
                f"buying precision here. Widen the {dominant} dtype first."
            ),
        )

    return RealizabilityVerdict(
        realizable=True,
        expected_improvement=None,
        dominant_surviving=dominant,
        explanation=(
            f"Realizable: the gradient's round-trip through {storage!r} "
            f"(unit roundoff {eps_grad:.2e}) dominates everything that "
            f"survives the fusion ({dominant} at {dom_eps:.2e}), so removing "
            f"it is the leading term. The improvement is large and its exact "
            f"size depends on gradient distribution and step count, which this "
            f"oracle does not model — measure it rather than quoting a factor."
        ),
    )
