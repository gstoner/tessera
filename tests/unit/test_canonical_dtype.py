"""Sprint A0 — canonical-dtype enforcement guard tests.

Locks the public contract established by
``docs/reference/tessera_tensor_attributes.md`` (normative, 2026-05-11):

  1. Canonical dtype set + aliases as enumerated in that doc.
  2. Planned/gated dtypes are recognized but *not* first-class.
  3. ``tf32`` is rejected as a storage dtype (it's a math_mode).
  4. ``canonicalize_dtype()`` normalizes aliases before storage.
  5. ``DistributedArray.from_domain`` / ``tessera.zeros``/etc. canonicalize
     at the API boundary.
  6. The standalone primitive coverage registry's
     ``assert_canonical_dtypes()`` walker is wired and passes.

If any of these regress, this test catches it before the new behaviour
ships.
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest

from tessera.dtype import (
    TesseraDtypeError,
    assert_canonical_dtype,
    canonical_dtypes,
    canonicalize_dtype,
    dtype_aliases,
    is_canonical_dtype,
    is_known_dtype,
    is_planned_gated_dtype,
    planned_gated_dtypes,
)


# ──────────────────────────────────────────────────────────────────────────
#                   Canonical / planned-gated tables
# ──────────────────────────────────────────────────────────────────────────

class TestCanonicalSet:
    def test_canonical_floats_present(self):
        c = canonical_dtypes()
        for name in ("fp64", "fp32", "fp16", "bf16"):
            assert name in c

    def test_canonical_low_precision_present(self):
        c = canonical_dtypes()
        for name in (
            "fp8_e4m3", "fp8_e5m2",
            "fp6_e2m3", "fp6_e3m2",
            "fp4_e2m1",
            "nvfp4",
        ):
            assert name in c

    def test_canonical_ints_present(self):
        c = canonical_dtypes()
        for name in ("int8", "int16", "int32", "int64"):
            assert name in c

    def test_canonical_bool_present(self):
        assert "bool" in canonical_dtypes()

    def test_unsigned_not_canonical(self):
        c = canonical_dtypes()
        for name in ("uint8", "uint16", "uint32", "uint64"):
            assert name not in c

    def test_complex_not_canonical(self):
        c = canonical_dtypes()
        for name in ("complex32", "complex64", "complex128"):
            assert name not in c


class TestPlannedGatedSet:
    def test_unsigned_planned_gated(self):
        for n in ("uint8", "uint16", "uint32", "uint64"):
            assert is_planned_gated_dtype(n), n

    def test_complex_planned_gated(self):
        for n in ("complex32", "complex64", "complex128"):
            assert is_planned_gated_dtype(n), n

    def test_packed_int4_planned_gated(self):
        assert is_planned_gated_dtype("int4")

    def test_amd_mx_planned_gated(self):
        for n in ("mxfp8", "mxfp6", "mxfp4"):
            assert is_planned_gated_dtype(n), n

    def test_canonical_and_planned_are_disjoint(self):
        assert canonical_dtypes() & planned_gated_dtypes() == frozenset()


# ──────────────────────────────────────────────────────────────────────────
#                          Alias normalization
# ──────────────────────────────────────────────────────────────────────────

class TestAliasNormalization:
    @pytest.mark.parametrize("alias,canonical", [
        ("f32", "fp32"),
        ("f16", "fp16"),
        ("f64", "fp64"),
        ("float32", "fp32"),
        ("float64", "fp64"),
        ("float16", "fp16"),
        ("bfloat16", "bf16"),
        ("fp8", "fp8_e4m3"),
        ("f8", "fp8_e4m3"),
        ("float8", "fp8_e4m3"),
        ("bf8", "fp8_e5m2"),
        ("bfloat8", "fp8_e5m2"),
        ("half", "fp16"),
        ("double", "fp64"),
        ("float", "fp32"),
        ("i8", "int8"),
        ("i16", "int16"),
        ("i32", "int32"),
        ("i64", "int64"),
        ("i1", "bool"),
        ("boolean", "bool"),
    ])
    def test_alias_maps_to_canonical(self, alias, canonical):
        assert canonicalize_dtype(alias) == canonical

    def test_case_insensitive_fold_to_canonical(self):
        assert canonicalize_dtype("FP32") == "fp32"
        assert canonicalize_dtype("BF16") == "bf16"
        assert canonicalize_dtype("INT32") == "int32"

    def test_canonical_round_trip(self):
        for name in canonical_dtypes():
            assert canonicalize_dtype(name) == name


# ──────────────────────────────────────────────────────────────────────────
#                              TF32 rejection
# ──────────────────────────────────────────────────────────────────────────

class TestTF32:
    def test_tf32_is_not_a_storage_dtype(self):
        for spelling in ("tf32", "TF32"):
            with pytest.raises(TesseraDtypeError, match="math_mode"):
                canonicalize_dtype(spelling)

    def test_tf32_is_not_canonical(self):
        assert not is_canonical_dtype("tf32")
        assert not is_planned_gated_dtype("tf32")

    def test_tf32_error_points_at_numeric_policy(self):
        with pytest.raises(TesseraDtypeError) as exc:
            canonicalize_dtype("tf32")
        assert "numeric_policy" in str(exc.value)


# ──────────────────────────────────────────────────────────────────────────
#                       Planned-gated acceptance gate
# ──────────────────────────────────────────────────────────────────────────

class TestPlannedGatedGate:
    @pytest.mark.parametrize("name", [
        "uint8", "uint16", "uint32", "uint64",
        "complex64", "complex128",
        "int4",
        "mxfp8", "mxfp6", "mxfp4",
    ])
    def test_planned_gated_rejected_without_flag(self, name):
        with pytest.raises(TesseraDtypeError, match="planned/gated"):
            canonicalize_dtype(name)

    @pytest.mark.parametrize("name", [
        "uint8", "complex64", "int4", "mxfp8",
    ])
    def test_planned_gated_accepted_with_flag(self, name):
        assert canonicalize_dtype(name, allow_planned_gated=True) == name


# ──────────────────────────────────────────────────────────────────────────
#                         Compound / invalid spellings
# ──────────────────────────────────────────────────────────────────────────

class TestRejection:
    def test_empty_dtype_rejected(self):
        with pytest.raises(TesseraDtypeError, match="empty"):
            canonicalize_dtype("")

    @pytest.mark.parametrize("compound", [
        "bf16/fp32",
        "fp16,fp32",
        "fp16+fp32",
    ])
    def test_compound_dtype_rejected(self, compound):
        with pytest.raises(TesseraDtypeError, match="numeric_policy"):
            canonicalize_dtype(compound)

    def test_unknown_dtype_rejected(self):
        with pytest.raises(TesseraDtypeError, match="unknown"):
            canonicalize_dtype("frobnicate")

    def test_non_string_rejected(self):
        with pytest.raises(TesseraDtypeError, match="string"):
            canonicalize_dtype(32)  # type: ignore[arg-type]


# ──────────────────────────────────────────────────────────────────────────
#                  Public API boundary — DistributedArray + factories
# ──────────────────────────────────────────────────────────────────────────

class TestApiBoundary:
    def test_zeros_canonicalizes_alias(self):
        import tessera as ts

        x = ts.zeros((4, 8), dtype="f32")
        assert x.dtype == "fp32"

    def test_ones_canonicalizes_bfloat16(self):
        import tessera as ts

        x = ts.ones((2, 2), dtype="bfloat16")
        assert x.dtype == "bf16"

    def test_randn_canonicalizes_half(self):
        import tessera as ts

        x = ts.randn((2,), dtype="half")
        assert x.dtype == "fp16"

    def test_zeros_rejects_tf32(self):
        import tessera as ts

        with pytest.raises(ValueError, match="math_mode"):
            ts.zeros((2,), dtype="tf32")

    def test_zeros_rejects_unknown_dtype(self):
        import tessera as ts

        with pytest.raises(ValueError, match="unknown"):
            ts.zeros((2,), dtype="frobnicate")

    def test_int_aliases_canonicalize(self):
        import tessera as ts

        for alias, canonical in (("i8", "int8"), ("i32", "int32"), ("i64", "int64")):
            x = ts.zeros((2,), dtype=alias)
            assert x.dtype == canonical, (alias, canonical, x.dtype)

    def test_parameter_inherits_canonicalization(self):
        from tessera.nn import Parameter

        # Parameter routes through DistributedArray.from_domain, which
        # canonicalizes.  Construct with an alias and confirm.
        p = Parameter(shape=(4, 4), dtype="f32")
        assert p.data.dtype == "fp32"


# ──────────────────────────────────────────────────────────────────────────
#               Registry walker — primitive_coverage.assert_canonical_dtypes
# ──────────────────────────────────────────────────────────────────────────

class TestRegistryDtypeAudit:
    def test_audit_buckets_present(self):
        from tessera.compiler.primitive_coverage import audit_canonical_dtypes

        buckets = audit_canonical_dtypes()
        assert set(buckets) == {"canonical", "alias", "planned_gated", "unknown"}

    def test_no_unknown_dtype_strings_in_registry(self):
        from tessera.compiler.primitive_coverage import audit_canonical_dtypes

        buckets = audit_canonical_dtypes()
        unknown = buckets["unknown"]
        assert not unknown, f"unknown dtype strings in registry: {unknown}"

    def test_assert_canonical_dtypes_passes(self):
        from tessera.compiler.primitive_coverage import assert_canonical_dtypes

        # Should be a no-op against the live registry (registry doesn't
        # currently store dtype strings); future schema additions are
        # gated by this walker.
        assert_canonical_dtypes()


# ──────────────────────────────────────────────────────────────────────────
#                   tessera.dtype module re-export
# ──────────────────────────────────────────────────────────────────────────

class TestPublicReexport:
    def test_tessera_dtype_module_importable(self):
        import tessera

        assert hasattr(tessera, "dtype")
        m = tessera.dtype
        assert callable(m.canonicalize_dtype)
        assert callable(m.is_canonical_dtype)
        assert callable(m.is_planned_gated_dtype)

    def test_assert_canonical_dtype_returns_canonical(self):
        assert assert_canonical_dtype("f32") == "fp32"
        assert assert_canonical_dtype("bf16") == "bf16"

    def test_assert_canonical_dtype_attaches_context(self):
        with pytest.raises(TesseraDtypeError, match="my_op input"):
            assert_canonical_dtype("frobnicate", context="my_op input")


# ──────────────────────────────────────────────────────────────────────────
#         Sprint F — public Dtype object + result_type lattice
# ──────────────────────────────────────────────────────────────────────────

from tessera.dtype import Dtype, result_type, canonicalize, is_canonical


class TestDtypeClass:
    def test_dtype_canonicalizes_alias(self):
        assert Dtype("f32") == "fp32"
        assert Dtype("bfloat16") == "bf16"
        assert Dtype("half") == "fp16"
        assert Dtype("i8") == "int8"

    def test_dtype_idempotent_on_dtype_input(self):
        d = Dtype("fp32")
        assert Dtype(d) is d
        assert Dtype.from_(d) is d

    def test_dtype_is_str_compatible(self):
        d = Dtype("fp16")
        assert isinstance(d, str)
        # Existing code paths that compare against string literals still work.
        assert d == "fp16"
        assert {d, "fp32"} == {"fp16", "fp32"}

    def test_dtype_repr_round_trip(self):
        d = Dtype("bf16")
        assert repr(d) == "Dtype('bf16')"

    def test_dtype_rejects_unknown(self):
        with pytest.raises(TesseraDtypeError):
            Dtype("frobnicate")

    def test_dtype_rejects_tf32(self):
        with pytest.raises(TesseraDtypeError, match="math_mode"):
            Dtype("tf32")

    def test_dtype_planned_gated_requires_flag(self):
        with pytest.raises(TesseraDtypeError, match="planned/gated"):
            Dtype("uint8")
        assert Dtype("uint8", allow_planned_gated=True) == "uint8"

    @pytest.mark.parametrize("name,expected", [
        ("fp64", 64), ("fp32", 32), ("fp16", 16), ("bf16", 16),
        ("fp8_e4m3", 8), ("fp8_e5m2", 8),
        ("fp6_e2m3", 6), ("fp6_e3m2", 6),
        ("fp4_e2m1", 4), ("nvfp4", 4),
        ("int8", 8), ("int16", 16), ("int32", 32), ("int64", 64),
        ("bool", 1),
    ])
    def test_dtype_bits(self, name, expected):
        assert Dtype(name).bits == expected

    def test_dtype_predicates(self):
        assert Dtype("fp32").is_float
        assert Dtype("bf16").is_float
        assert Dtype("nvfp4").is_low_precision
        assert Dtype("nvfp4").is_float  # low-precision floats are still floats
        assert Dtype("int32").is_integer
        assert not Dtype("int32").is_float
        assert not Dtype("fp32").is_integer

    def test_dtype_canonical_classmethods(self):
        assert Dtype.canonical("f32") == "fp32"
        # Roundtrip via from_
        d = Dtype("bf16")
        assert Dtype.from_(d) is d
        assert Dtype.from_("fp32") == "fp32"


class TestResultType:
    """Standard NumPy/JAX-style implicit promotion."""

    @pytest.mark.parametrize("a,b,expected", [
        ("fp16", "fp32", "fp32"),
        ("fp32", "fp16", "fp32"),
        ("fp64", "fp16", "fp64"),
        ("fp32", "fp64", "fp64"),
        # bf16/fp16 incomparable → fp32 safety net
        ("bf16", "fp16", "fp32"),
        ("fp16", "bf16", "fp32"),
        # Identity
        ("fp32", "fp32", "fp32"),
        ("bf16", "bf16", "bf16"),
        # Int + int
        ("int32", "int64", "int64"),
        ("int8", "int16", "int16"),
        ("bool", "int8", "int8"),
        # Float + int → float, with safety net for wide ints + narrow floats
        ("int8", "fp16", "fp16"),
        ("int8", "bf16", "bf16"),
        ("int32", "fp16", "fp32"),
        ("int64", "bf16", "fp32"),
        ("int32", "fp32", "fp32"),
        ("int64", "fp64", "fp64"),
        # Low-precision routes through fp32 family
        ("fp8_e4m3", "fp16", "fp32"),
        ("fp8_e4m3", "fp32", "fp32"),
        ("fp8_e4m3", "fp64", "fp64"),
        ("nvfp4", "bf16", "fp32"),
        ("fp4_e2m1", "fp8_e4m3", "fp32"),
    ])
    def test_pairwise_promotion(self, a, b, expected):
        assert result_type(a, b) == expected
        # Commutative
        assert result_type(b, a) == expected

    def test_returns_dtype_instance(self):
        out = result_type("fp16", "fp32")
        assert isinstance(out, Dtype)

    def test_n_ary_promotion(self):
        # left-fold over the list
        assert result_type("int8", "fp16", "fp32") == "fp32"
        assert result_type("bool", "int8", "int16", "int32") == "int32"

    def test_aliases_canonicalize_before_promotion(self):
        assert result_type("f16", "f32") == "fp32"
        assert result_type("bfloat16", "half") == "fp32"

    def test_single_arg_returns_canonical(self):
        assert result_type("f32") == "fp32"
        assert isinstance(result_type("f32"), Dtype)

    def test_zero_args_rejected(self):
        with pytest.raises(TesseraDtypeError, match="at least one"):
            result_type()

    def test_invalid_mode_rejected(self):
        with pytest.raises(TesseraDtypeError, match="must be 'standard' or 'strict'"):
            result_type("fp32", "fp32", mode="bogus")

    def test_tf32_rejected_in_promotion(self):
        with pytest.raises(TesseraDtypeError):
            result_type("fp32", "tf32")


class TestResultTypeStrict:
    """Strict mode rejects mixed dtypes."""

    def test_strict_same_dtype_passes(self):
        assert result_type("fp32", "fp32", "fp32", mode="strict") == "fp32"
        assert result_type("bf16", "bf16", mode="strict") == "bf16"

    def test_strict_alias_normalization_still_works(self):
        # f32 + fp32 are the same canonical dtype after normalization.
        assert result_type("f32", "fp32", mode="strict") == "fp32"

    @pytest.mark.parametrize("a,b", [
        ("fp32", "fp16"),
        ("bf16", "fp16"),
        ("int32", "fp32"),
        ("fp8_e4m3", "bf16"),
    ])
    def test_strict_mixed_dtypes_rejected(self, a, b):
        with pytest.raises(TesseraDtypeError, match="strict-mode result_type rejects"):
            result_type(a, b, mode="strict")


class TestDtypeOperators:
    def test_or_operator_is_promotion(self):
        assert (Dtype("fp16") | Dtype("bf16")) == "fp32"
        assert (Dtype("int8") | Dtype("fp16")) == "fp16"
        assert (Dtype("int32") | Dtype("fp64")) == "fp64"

    def test_or_with_string_works(self):
        assert (Dtype("fp16") | "fp32") == "fp32"
        assert ("fp16" | Dtype("fp32")) == "fp32"


class TestShortAliases:
    def test_canonicalize_short_alias_is_canonicalize_dtype(self):
        from tessera.dtype import canonicalize_dtype
        assert canonicalize is canonicalize_dtype

    def test_is_canonical_short_alias(self):
        from tessera.dtype import is_canonical_dtype
        assert is_canonical is is_canonical_dtype

    def test_is_planned_gated_short_alias(self):
        from tessera.dtype import is_planned_gated, is_planned_gated_dtype
        assert is_planned_gated is is_planned_gated_dtype
