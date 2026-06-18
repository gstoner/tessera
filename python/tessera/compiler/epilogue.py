"""Composable bit-flag epilogue specification (hipBLASLt-modeled).

AMD hipBLASLt exposes a single ``hipblasLtEpilogue_t`` bit-flag enum that
selects the *epilogue* fused onto a GEMM — the activation / bias / aux-output
stage that runs in the matmul's output registers before the result is written
to memory.  The flag values are deliberately chosen so that the *forward*
epilogues compose by bitwise OR: ``RELU | BIAS`` literally equals the
``RELU_BIAS`` flag value, and ``GELU_AUX | BIAS`` equals ``GELU_AUX_BIAS``.

This module ports that bit-flag vocabulary into Tessera as a compiler-visible
contract — pure data, no kernel import.  It is a **leaf module** (depends only
on the stdlib — ``enum.IntFlag`` + ``dataclasses``) so the audit registry
(``primitive_coverage``), ``op_catalog``, the backend manifest, and the runtime
can all import it without a cycle.

The *killer feature* is the autodiff bridge.  A fused matmul epilogue can store
its **pre-activation** tensor — the ``A@B (+bias)`` value *before* the
activation is applied — into an auxiliary (``*_AUX``) output.  The backward
epilogues (``DRELU`` / ``DGELU`` / ``BGRAD*``) consume exactly that aux tensor.
This is what makes the ``*_AUX`` forward epilogues and the ``D{RELU,GELU}`` /
``BGRAD*`` backward epilogues **autodiff primitives**:

    A fused matmul's VJP requests the ``*_AUX`` pre-activation tensor that the
    forward pass already stored, rather than recomputing ``A@B`` from scratch.

So a VJP/JVP registry entry for a fused matmul epilogue uses
:func:`backward_epilogue` to discover which backward epilogue consumes its aux,
and :func:`requires_aux` to decide whether the forward pass must materialize the
aux tensor (it does whenever the activation's derivative cannot be reconstructed
from the activated output alone — i.e. for gelu/sigmoid/silu, and for relu since
the sign mask is cheapest to read from the stored pre-activation).
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Any

__all__ = [
    "Epilogue",
    "EpilogueSpec",
    "backward_epilogue",
    "requires_aux",
    "CANONICAL_EPILOGUES",
    "ACTIVATION_NOT_IMPLEMENTED",
]


class Epilogue(enum.IntFlag):
    """hipBLASLt ``hipblasLtEpilogue_t`` epilogue selectors (OR-composable).

    The values match AMD's enum exactly so the forward epilogues compose by
    bitwise OR — ``Epilogue.RELU | Epilogue.BIAS == Epilogue.RELU_BIAS`` (6),
    ``Epilogue.GELU | Epilogue.BIAS == Epilogue.GELU_BIAS`` (36), and
    ``Epilogue.GELU_AUX | Epilogue.BIAS == Epilogue.GELU_AUX_BIAS`` (164).

    Bit decomposition (the structure the OR-composition relies on):

    * ``NONE``        = 1        — identity / default
    * ``RELU``        = 2        — relu activation bit
    * ``BIAS``        = 4        — fused bias-add bit
    * ``GELU``        = 32       — gelu activation bit
    * ``*_AUX``       = + 128    — also store the pre-activation aux tensor
    * ``D{RELU,GELU}``           — backward activation epilogues
    * ``BGRAD{A,B}``  = 256/512  — accumulate the bias gradient over A / B
    * ``SIGMOID``     = 1024     — sigmoid activation
    * ``SWISH_EXT``   = 65536    — SiLU / swish activation (extension)
    * ``CLAMP_EXT``   = 131072   — clamp activation (extension)
    """

    NONE = 1
    DEFAULT = 1
    RELU = 2
    BIAS = 4
    RELU_BIAS = 6
    GELU = 32
    GELU_BIAS = 36
    RELU_AUX = 130
    RELU_AUX_BIAS = 134
    GELU_AUX = 160
    GELU_AUX_BIAS = 164
    DGELU = 192
    DGELU_BGRAD = 208
    DRELU = 136
    DRELU_BGRAD = 152
    BGRADA = 256
    BGRADB = 512
    SIGMOID = 1024
    SWISH_EXT = 65536  # SiLU / swish
    CLAMP_EXT = 131072


# Bit masks used by the predicates below.  ``_AUX_BIT`` marks "also emit the
# pre-activation aux tensor"; ``_BIAS_BIT`` marks the fused bias-add.
_AUX_BIT = Epilogue(128)
_BIAS_BIT = Epilogue.BIAS

# Forward activation bits (an epilogue "has an activation" if any of these set).
_ACTIVATION_BITS = (
    Epilogue.RELU,
    Epilogue.GELU,
    Epilogue.SIGMOID,
    Epilogue.SWISH_EXT,
    Epilogue.CLAMP_EXT,
)

# Backward epilogues (D-prefixed activation backward + standalone bias-grad).
_BACKWARD_FLAGS = (
    Epilogue.DRELU,
    Epilogue.DRELU_BGRAD,
    Epilogue.DGELU,
    Epilogue.DGELU_BGRAD,
    Epilogue.BGRADA,
    Epilogue.BGRADB,
)

# Sentinel returned by ``backward_epilogue`` when the caller opts into a
# documented "no backward registered" sentinel instead of an exception.
ACTIVATION_NOT_IMPLEMENTED = "not_implemented"


def _has_bit(flags: Epilogue, bit: Epilogue) -> bool:
    """True iff *all* bits of ``bit`` are set in ``flags`` (exact submask)."""
    return (flags & bit) == bit


@dataclass(frozen=True)
class EpilogueSpec:
    """A fully-resolved epilogue: the bit flags plus their parameters.

    ``flags`` selects the fused epilogue; the optional fields carry the
    parameters those flags need:

    * ``clamp_lo`` / ``clamp_hi`` — bounds for ``CLAMP_EXT`` (clamp to
      ``[clamp_lo, clamp_hi]``).
    * ``aux_ld`` — leading dimension (stride) for the ``*_AUX`` output tensor.
    * ``scale`` — optional output scale applied in the epilogue.
    """

    flags: Epilogue
    clamp_lo: float | None = None
    clamp_hi: float | None = None
    aux_ld: int | None = None
    scale: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.flags, Epilogue):
            raise ValueError(
                f"flags must be an Epilogue bit-flag; got {type(self.flags).__name__}")
        if int(self.flags) <= 0:
            raise ValueError(f"flags must be a positive Epilogue value; got {self.flags!r}")

        # A forward activation cannot be combined with a backward (D*) flag —
        # the backward epilogue *consumes* a forward activation's aux, it does
        # not run alongside one.
        if self.has_activation and self.is_backward:
            raise ValueError(
                f"cannot combine a forward activation with a backward epilogue: "
                f"{self.flags!r}")

        # CLAMP_EXT needs valid bounds.
        if _has_bit(self.flags, Epilogue.CLAMP_EXT):
            if self.clamp_lo is None or self.clamp_hi is None:
                raise ValueError(
                    "CLAMP_EXT requires both clamp_lo and clamp_hi")
            if self.clamp_lo > self.clamp_hi:
                raise ValueError(
                    f"CLAMP_EXT requires clamp_lo <= clamp_hi; got "
                    f"clamp_lo={self.clamp_lo}, clamp_hi={self.clamp_hi}")
        elif self.clamp_lo is not None or self.clamp_hi is not None:
            raise ValueError(
                "clamp_lo/clamp_hi are only valid when CLAMP_EXT is set")

        if self.aux_ld is not None and self.aux_ld <= 0:
            raise ValueError(f"aux_ld must be positive; got {self.aux_ld}")

    # ── activation predicates ────────────────────────────────────────────────
    @property
    def has_activation(self) -> bool:
        """True if any forward activation bit (relu/gelu/sigmoid/silu/clamp) is set."""
        return any(_has_bit(self.flags, bit) for bit in _ACTIVATION_BITS)

    @property
    def activation_kind(self) -> str | None:
        """Canonical activation name, or ``None`` if no forward activation."""
        if _has_bit(self.flags, Epilogue.GELU):
            return "gelu"
        if _has_bit(self.flags, Epilogue.RELU):
            return "relu"
        if _has_bit(self.flags, Epilogue.SWISH_EXT):
            return "silu"
        if _has_bit(self.flags, Epilogue.SIGMOID):
            return "sigmoid"
        if _has_bit(self.flags, Epilogue.CLAMP_EXT):
            return "clamp"
        return None

    # ── bias predicates ──────────────────────────────────────────────────────
    @property
    def has_bias(self) -> bool:
        """True if the fused bias-add bit is set."""
        return _has_bit(self.flags, _BIAS_BIT)

    @property
    def bias_grad_operand(self) -> str | None:
        """Operand the bias gradient accumulates over: ``"A"`` / ``"B"`` / None."""
        if _has_bit(self.flags, Epilogue.BGRADA):
            return "A"
        if _has_bit(self.flags, Epilogue.BGRADB):
            return "B"
        return None

    # ── aux / autodiff predicates ────────────────────────────────────────────
    @property
    def has_aux(self) -> bool:
        """True if this epilogue stores the pre-activation aux tensor (``*_AUX``).

        The aux tensor holds ``A@B (+bias)`` *before* the activation — exactly
        what the backward epilogue needs to compute the activation's gradient
        without recomputing the matmul.
        """
        return _has_bit(self.flags, _AUX_BIT) and not self.is_backward

    @property
    def is_backward(self) -> bool:
        """True if this is a backward epilogue (D-prefixed or BGRAD-only)."""
        return any(_has_bit(self.flags, bw) for bw in _BACKWARD_FLAGS)

    # ── serialization ────────────────────────────────────────────────────────
    def as_metadata_dict(self) -> dict[str, Any]:
        return {
            "flags": int(self.flags),
            "flags_name": self.flags.name if self.flags.name is not None
            else "|".join(f.name for f in Epilogue if f & self.flags and f.name),
            "has_activation": self.has_activation,
            "activation_kind": self.activation_kind,
            "has_bias": self.has_bias,
            "has_aux": self.has_aux,
            "is_backward": self.is_backward,
            "bias_grad_operand": self.bias_grad_operand,
            "clamp_lo": self.clamp_lo,
            "clamp_hi": self.clamp_hi,
            "aux_ld": self.aux_ld,
            "scale": self.scale,
        }

    @classmethod
    def from_metadata_dict(cls, d: dict[str, Any]) -> "EpilogueSpec":
        """Inverse of :meth:`as_metadata_dict` (round-trips ``flags`` + params)."""
        return cls(
            flags=Epilogue(int(d["flags"])),
            clamp_lo=d.get("clamp_lo"),
            clamp_hi=d.get("clamp_hi"),
            aux_ld=d.get("aux_ld"),
            scale=d.get("scale"),
        )


# ── Autodiff bridge — the killer feature ─────────────────────────────────────
# Maps a forward activation to the backward activation epilogue that consumes
# its aux (pre-activation) tensor.  ``None`` marks "no backward registered".
_BACKWARD_ACTIVATION: dict[str, Epilogue | None] = {
    "relu": Epilogue.DRELU,
    "gelu": Epilogue.DGELU,
    "sigmoid": None,  # backward not implemented in hipBLASLt
    "silu": None,     # backward not implemented in hipBLASLt
    "clamp": None,    # backward not implemented in hipBLASLt
}

# When a forward epilogue also fused a bias, the backward must accumulate the
# bias gradient — these are the ``*_BGRAD`` backward variants.
_BACKWARD_ACTIVATION_BGRAD: dict[str, Epilogue] = {
    "relu": Epilogue.DRELU_BGRAD,
    "gelu": Epilogue.DGELU_BGRAD,
}


def backward_epilogue(forward: EpilogueSpec,
                      *, error_on_unimplemented: bool = True) -> EpilogueSpec:
    """Map a forward fused epilogue to its backward (gradient) epilogue.

    The backward epilogue consumes the forward pass's stored ``*_AUX``
    pre-activation tensor:

    * forward ``GELU`` / ``GELU_AUX``     → backward ``DGELU``
    * forward ``GELU_BIAS`` / ``GELU_AUX_BIAS`` → backward ``DGELU_BGRAD``
    * forward ``RELU`` / ``RELU_AUX``     → backward ``DRELU``
    * forward ``RELU_BIAS`` / ``RELU_AUX_BIAS`` → backward ``DRELU_BGRAD``

    This is the function a fused-matmul VJP calls to discover which backward
    epilogue to launch (and, via :func:`requires_aux`, whether the forward must
    materialize the aux tensor).

    ``sigmoid`` / ``silu`` / ``clamp`` have no backward epilogue in the
    hipBLASLt vocabulary.  By default this raises ``ValueError`` naming the
    activation; pass ``error_on_unimplemented=False`` to instead raise a
    ``ValueError`` whose message is exactly the sentinel string
    :data:`ACTIVATION_NOT_IMPLEMENTED`, so callers can branch on it cleanly.
    """
    if forward.is_backward:
        raise ValueError(
            f"backward_epilogue expects a forward epilogue; got backward "
            f"{forward.flags!r}")
    kind = forward.activation_kind
    if kind is None:
        raise ValueError(
            f"backward_epilogue requires a forward activation; {forward.flags!r} "
            f"has none")

    bw = _BACKWARD_ACTIVATION.get(kind)
    if bw is None:
        if error_on_unimplemented:
            raise ValueError(
                f"no backward epilogue registered for activation {kind!r} "
                f"(hipBLASLt provides backward only for relu/gelu); "
                f"sentinel = {ACTIVATION_NOT_IMPLEMENTED!r}")
        # Documented sentinel path: a NONE spec is meaningless here, so we still
        # cannot synthesize a real backward — re-raise with the sentinel name.
        raise ValueError(ACTIVATION_NOT_IMPLEMENTED)

    if forward.has_bias:
        bw = _BACKWARD_ACTIVATION_BGRAD[kind]

    # Carry forward the aux leading dim so the backward reads the same buffer.
    return EpilogueSpec(flags=bw, aux_ld=forward.aux_ld)


def requires_aux(forward: EpilogueSpec) -> bool:
    """True if the forward pass must store the ``*_AUX`` pre-activation tensor.

    The backward epilogue needs the pre-activation whenever the activation's
    derivative cannot be reconstructed from the *activated* output alone:

    * ``gelu`` / ``sigmoid`` / ``silu`` — non-trivial derivatives, definitely
      need the pre-activation.
    * ``relu`` — its backward only needs the sign mask, which the stored
      pre-activation also provides (and is the form the ``DRELU`` epilogue
      reads), so this returns ``True`` for relu too.

    Returns ``False`` for a pure bias / scale / clamp / no-activation epilogue
    (clamp has no registered backward, so nothing requests its aux).
    """
    kind = forward.activation_kind
    return kind in ("relu", "gelu", "sigmoid", "silu")


# ── Canonical fused epilogues used by the backends ───────────────────────────
CANONICAL_EPILOGUES: dict[str, EpilogueSpec] = {
    "matmul_relu": EpilogueSpec(flags=Epilogue.RELU),
    "matmul_gelu": EpilogueSpec(flags=Epilogue.GELU),
    "matmul_bias": EpilogueSpec(flags=Epilogue.BIAS),
    "matmul_bias_gelu": EpilogueSpec(flags=Epilogue.GELU_BIAS),
    "matmul_silu": EpilogueSpec(flags=Epilogue.SWISH_EXT),
    "matmul_sigmoid": EpilogueSpec(flags=Epilogue.SIGMOID),
}
