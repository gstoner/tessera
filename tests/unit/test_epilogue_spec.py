"""Tests for the composable bit-flag epilogue spec (hipBLASLt-modeled)."""

from __future__ import annotations

import pytest

from tessera.compiler.epilogue import (
    ACTIVATION_NOT_IMPLEMENTED,
    CANONICAL_EPILOGUES,
    Epilogue,
    EpilogueSpec,
    backward_epilogue,
    requires_aux,
)


# ── exact bit values + OR-composition ────────────────────────────────────────
def test_exact_bit_values() -> None:
    assert int(Epilogue.NONE) == 1
    assert int(Epilogue.DEFAULT) == 1
    assert int(Epilogue.RELU) == 2
    assert int(Epilogue.BIAS) == 4
    assert int(Epilogue.RELU_BIAS) == 6
    assert int(Epilogue.GELU) == 32
    assert int(Epilogue.GELU_BIAS) == 36
    assert int(Epilogue.RELU_AUX) == 130
    assert int(Epilogue.RELU_AUX_BIAS) == 134
    assert int(Epilogue.GELU_AUX) == 160
    assert int(Epilogue.GELU_AUX_BIAS) == 164
    assert int(Epilogue.DGELU) == 192
    assert int(Epilogue.DGELU_BGRAD) == 208
    assert int(Epilogue.DRELU) == 136
    assert int(Epilogue.DRELU_BGRAD) == 152
    assert int(Epilogue.BGRADA) == 256
    assert int(Epilogue.BGRADB) == 512
    assert int(Epilogue.SIGMOID) == 1024
    assert int(Epilogue.SWISH_EXT) == 65536
    assert int(Epilogue.CLAMP_EXT) == 131072


def test_or_composition() -> None:
    # The headline property: forward epilogue flags compose by bitwise OR.
    assert (Epilogue.RELU | Epilogue.BIAS) == Epilogue.RELU_BIAS
    assert int(Epilogue.RELU | Epilogue.BIAS) == 6
    assert (Epilogue.GELU | Epilogue.BIAS) == Epilogue.GELU_BIAS
    assert int(Epilogue.GELU | Epilogue.BIAS) == 36
    assert (Epilogue.GELU_AUX | Epilogue.BIAS) == Epilogue.GELU_AUX_BIAS
    assert int(Epilogue.GELU_AUX | Epilogue.BIAS) == 164
    # AUX bit composes with the activation: GELU | 128 == GELU_AUX.
    assert (Epilogue.GELU | Epilogue(128)) == Epilogue.GELU_AUX


# ── activation predicates ────────────────────────────────────────────────────
def test_activation_predicates() -> None:
    assert EpilogueSpec(flags=Epilogue.RELU).activation_kind == "relu"
    assert EpilogueSpec(flags=Epilogue.GELU).activation_kind == "gelu"
    assert EpilogueSpec(flags=Epilogue.SIGMOID).activation_kind == "sigmoid"
    assert EpilogueSpec(flags=Epilogue.SWISH_EXT).activation_kind == "silu"
    assert EpilogueSpec(flags=Epilogue.NONE).activation_kind is None
    assert EpilogueSpec(flags=Epilogue.BIAS).activation_kind is None

    clamp = EpilogueSpec(flags=Epilogue.CLAMP_EXT, clamp_lo=-1.0, clamp_hi=1.0)
    assert clamp.activation_kind == "clamp"
    assert clamp.has_activation

    assert EpilogueSpec(flags=Epilogue.RELU_BIAS).has_activation
    assert not EpilogueSpec(flags=Epilogue.BIAS).has_activation
    # AUX activation still reports its activation kind.
    assert EpilogueSpec(flags=Epilogue.GELU_AUX).activation_kind == "gelu"


# ── bias predicates ──────────────────────────────────────────────────────────
def test_bias_predicates() -> None:
    assert EpilogueSpec(flags=Epilogue.BIAS).has_bias
    assert EpilogueSpec(flags=Epilogue.RELU_BIAS).has_bias
    assert EpilogueSpec(flags=Epilogue.GELU_AUX_BIAS).has_bias
    assert not EpilogueSpec(flags=Epilogue.RELU).has_bias

    assert EpilogueSpec(flags=Epilogue.BGRADA).bias_grad_operand == "A"
    assert EpilogueSpec(flags=Epilogue.BGRADB).bias_grad_operand == "B"
    assert EpilogueSpec(flags=Epilogue.RELU_BIAS).bias_grad_operand is None


# ── aux / backward predicates ────────────────────────────────────────────────
def test_aux_predicates() -> None:
    assert EpilogueSpec(flags=Epilogue.RELU_AUX).has_aux
    assert EpilogueSpec(flags=Epilogue.GELU_AUX).has_aux
    assert EpilogueSpec(flags=Epilogue.GELU_AUX_BIAS).has_aux
    assert not EpilogueSpec(flags=Epilogue.GELU).has_aux
    assert not EpilogueSpec(flags=Epilogue.RELU).has_aux


def test_backward_predicates() -> None:
    assert EpilogueSpec(flags=Epilogue.DRELU).is_backward
    assert EpilogueSpec(flags=Epilogue.DGELU).is_backward
    assert EpilogueSpec(flags=Epilogue.DGELU_BGRAD).is_backward
    assert EpilogueSpec(flags=Epilogue.BGRADA).is_backward
    assert EpilogueSpec(flags=Epilogue.BGRADB).is_backward
    assert not EpilogueSpec(flags=Epilogue.GELU).is_backward
    # Backward epilogues do not report has_aux (they consume, not produce).
    assert not EpilogueSpec(flags=Epilogue.DGELU).has_aux


# ── autodiff bridge: forward -> backward ─────────────────────────────────────
def test_backward_epilogue_mapping() -> None:
    assert backward_epilogue(EpilogueSpec(flags=Epilogue.GELU)).flags == Epilogue.DGELU
    assert backward_epilogue(EpilogueSpec(flags=Epilogue.GELU_AUX)).flags == Epilogue.DGELU
    assert backward_epilogue(
        EpilogueSpec(flags=Epilogue.GELU_BIAS)).flags == Epilogue.DGELU_BGRAD
    assert backward_epilogue(
        EpilogueSpec(flags=Epilogue.GELU_AUX_BIAS)).flags == Epilogue.DGELU_BGRAD
    assert backward_epilogue(EpilogueSpec(flags=Epilogue.RELU)).flags == Epilogue.DRELU
    assert backward_epilogue(EpilogueSpec(flags=Epilogue.RELU_AUX)).flags == Epilogue.DRELU
    assert backward_epilogue(
        EpilogueSpec(flags=Epilogue.RELU_BIAS)).flags == Epilogue.DRELU_BGRAD


def test_backward_epilogue_carries_aux_ld() -> None:
    fwd = EpilogueSpec(flags=Epilogue.GELU_AUX, aux_ld=4096)
    assert backward_epilogue(fwd).aux_ld == 4096


def test_backward_epilogue_unimplemented_raises() -> None:
    with pytest.raises(ValueError, match="silu"):
        backward_epilogue(EpilogueSpec(flags=Epilogue.SWISH_EXT))
    with pytest.raises(ValueError, match="sigmoid"):
        backward_epilogue(EpilogueSpec(flags=Epilogue.SIGMOID))


def test_backward_epilogue_unimplemented_sentinel() -> None:
    with pytest.raises(ValueError, match=ACTIVATION_NOT_IMPLEMENTED):
        backward_epilogue(EpilogueSpec(flags=Epilogue.SIGMOID),
                          error_on_unimplemented=False)


def test_backward_epilogue_rejects_backward_input() -> None:
    with pytest.raises(ValueError, match="forward epilogue"):
        backward_epilogue(EpilogueSpec(flags=Epilogue.DGELU))


def test_backward_epilogue_requires_activation() -> None:
    with pytest.raises(ValueError, match="requires a forward activation"):
        backward_epilogue(EpilogueSpec(flags=Epilogue.BIAS))


# ── requires_aux ─────────────────────────────────────────────────────────────
def test_requires_aux() -> None:
    assert requires_aux(EpilogueSpec(flags=Epilogue.GELU))
    assert requires_aux(EpilogueSpec(flags=Epilogue.SIGMOID))
    assert requires_aux(EpilogueSpec(flags=Epilogue.SWISH_EXT))
    assert requires_aux(EpilogueSpec(flags=Epilogue.RELU))  # sign mask via aux
    assert not requires_aux(EpilogueSpec(flags=Epilogue.BIAS))
    assert not requires_aux(
        EpilogueSpec(flags=Epilogue.CLAMP_EXT, clamp_lo=0.0, clamp_hi=1.0))


# ── validation errors ────────────────────────────────────────────────────────
def test_clamp_requires_bounds() -> None:
    with pytest.raises(ValueError, match="CLAMP_EXT requires both"):
        EpilogueSpec(flags=Epilogue.CLAMP_EXT)


def test_clamp_bounds_ordering() -> None:
    with pytest.raises(ValueError, match="clamp_lo <= clamp_hi"):
        EpilogueSpec(flags=Epilogue.CLAMP_EXT, clamp_lo=1.0, clamp_hi=-1.0)
    # equal bounds are allowed
    EpilogueSpec(flags=Epilogue.CLAMP_EXT, clamp_lo=0.5, clamp_hi=0.5)


def test_clamp_bounds_without_flag_rejected() -> None:
    with pytest.raises(ValueError, match="only valid when CLAMP_EXT"):
        EpilogueSpec(flags=Epilogue.RELU, clamp_lo=0.0, clamp_hi=1.0)


def test_forward_activation_plus_backward_rejected() -> None:
    with pytest.raises(ValueError, match="forward activation with a backward"):
        EpilogueSpec(flags=Epilogue.RELU | Epilogue.DGELU)


def test_bad_flags_type() -> None:
    with pytest.raises(ValueError, match="must be an Epilogue"):
        EpilogueSpec(flags=6)  # type: ignore[arg-type]


def test_bad_aux_ld() -> None:
    with pytest.raises(ValueError, match="aux_ld must be positive"):
        EpilogueSpec(flags=Epilogue.GELU_AUX, aux_ld=0)


# ── metadata round-trip ──────────────────────────────────────────────────────
def test_metadata_round_trip() -> None:
    spec = EpilogueSpec(flags=Epilogue.GELU_AUX_BIAS, aux_ld=2048, scale=0.5)
    md = spec.as_metadata_dict()
    assert md["flags"] == int(Epilogue.GELU_AUX_BIAS)
    assert md["has_activation"] is True
    assert md["activation_kind"] == "gelu"
    assert md["has_bias"] is True
    assert md["has_aux"] is True
    assert md["is_backward"] is False
    assert md["aux_ld"] == 2048
    assert md["scale"] == 0.5
    assert EpilogueSpec.from_metadata_dict(md) == spec


def test_metadata_round_trip_clamp() -> None:
    spec = EpilogueSpec(flags=Epilogue.CLAMP_EXT, clamp_lo=-3.0, clamp_hi=3.0)
    md = spec.as_metadata_dict()
    assert md["activation_kind"] == "clamp"
    assert md["clamp_lo"] == -3.0
    assert md["clamp_hi"] == 3.0
    assert EpilogueSpec.from_metadata_dict(md) == spec


# ── canonical catalog ────────────────────────────────────────────────────────
def test_canonical_catalog_keys() -> None:
    for key in ("matmul_relu", "matmul_gelu", "matmul_bias",
                "matmul_bias_gelu", "matmul_silu", "matmul_sigmoid"):
        assert key in CANONICAL_EPILOGUES
        assert isinstance(CANONICAL_EPILOGUES[key], EpilogueSpec)


def test_canonical_catalog_predicates() -> None:
    assert CANONICAL_EPILOGUES["matmul_relu"].activation_kind == "relu"
    assert CANONICAL_EPILOGUES["matmul_gelu"].activation_kind == "gelu"
    assert CANONICAL_EPILOGUES["matmul_silu"].activation_kind == "silu"
    assert CANONICAL_EPILOGUES["matmul_sigmoid"].activation_kind == "sigmoid"

    assert CANONICAL_EPILOGUES["matmul_bias"].has_bias
    assert not CANONICAL_EPILOGUES["matmul_bias"].has_activation
    assert CANONICAL_EPILOGUES["matmul_bias_gelu"].has_bias
    assert CANONICAL_EPILOGUES["matmul_bias_gelu"].activation_kind == "gelu"
    # bias_gelu maps to the bias-grad backward.
    assert backward_epilogue(
        CANONICAL_EPILOGUES["matmul_bias_gelu"]).flags == Epilogue.DGELU_BGRAD
