"""Canonical dtype names + aliases + planned/gated set for Tessera.

This module is the **single source of truth** for the dtype string vocabulary
used across the registry (`compiler/primitive_coverage.py`), the public Python
API (`DistributedArray`, `Parameter`, `tessera.zeros`/`ones`/`randn`/etc.),
the JIT path, and serialized IR metadata.

Reference: ``docs/reference/tessera_tensor_attributes.md`` (normative,
2026-05-11) — the user-facing tensor attribute spec.

Rules locked here:

1. **Canonical names are the spelling stored in IR metadata.** When a
   non-canonical alias arrives at the API boundary (e.g., ``"f32"``), it is
   normalized to ``"fp32"`` before storage. ``canonicalize_dtype()`` performs
   the normalization.

2. **TF32 is not a storage dtype.** It is modelled as ``math_mode="tf32"`` on
   ``fp32`` tensors via ``numeric_policy``. ``canonicalize_dtype("tf32")``
   raises ``TesseraDtypeError`` — use ``NumericPolicy(storage="fp32",
   math_mode="tf32")`` instead.

3. **Planned/gated dtypes are not first-class today.** ``uint8/16/32/64``,
   ``complex64/128``, packed ``int4``, AMD ``mxfp8/6/4``, and Tenstorrent
   ``bfp8/4`` / ``blockfp8/4`` are recognized by ``is_planned_gated_dtype``
   but ``canonicalize_dtype()`` will raise unless ``allow_planned_gated=True``
   is passed. Registry entries that reference these must declare
   ``metadata.dtype_status = "planned_gated"``.

4. **Storage dtype only.** Accumulator dtype, rounding mode, quantization
   scale/axis, and determinism flags belong on ``numeric_policy``, not on
   the dtype string. ``canonicalize_dtype()`` rejects compound spellings
   like ``"bf16/fp32"``.

Public surface this module exposes:

    canonicalize_dtype(s, *, allow_planned_gated=False) -> str
    is_canonical_dtype(s) -> bool
    is_planned_gated_dtype(s) -> bool
    is_known_dtype(s) -> bool          # canonical OR planned-gated OR alias
    canonical_dtypes()  -> frozenset[str]
    planned_gated_dtypes() -> frozenset[str]
    dtype_aliases()     -> dict[str, str]
    TesseraDtypeError   (subclass of ValueError)
"""

from __future__ import annotations


class TesseraDtypeError(ValueError):
    """Raised when a dtype string is not canonical / not normalizable.

    Subclasses ``ValueError`` so existing ``except ValueError`` blocks in the
    runtime keep working unchanged.
    """


# ──────────────────────────────────────────────────────────────────────────
# Canonical dtype set (enumerated from tessera_tensor_attributes.md)
# ──────────────────────────────────────────────────────────────────────────

_CANONICAL_FLOATS: frozenset[str] = frozenset({
    "fp64", "fp32", "fp16", "bf16",
})

_CANONICAL_LOW_PRECISION: frozenset[str] = frozenset({
    "fp8_e4m3", "fp8_e5m2",
    "fp6_e2m3", "fp6_e3m2",
    "fp4_e2m1",
    "nvfp4",  # NVIDIA block-scaled FP4; distinct from OCP FP4 / AMD MXFP4
})

_CANONICAL_INTS: frozenset[str] = frozenset({
    "int8", "int16", "int32", "int64",
})

_CANONICAL_BOOL: frozenset[str] = frozenset({"bool"})

_CANONICAL_DTYPES: frozenset[str] = (
    _CANONICAL_FLOATS
    | _CANONICAL_LOW_PRECISION
    | _CANONICAL_INTS
    | _CANONICAL_BOOL
)


# ──────────────────────────────────────────────────────────────────────────
# Planned / gated dtype set (recognized but not first-class today)
# ──────────────────────────────────────────────────────────────────────────

_PLANNED_UNSIGNED: frozenset[str] = frozenset({
    "uint8", "uint16", "uint32", "uint64",
})

_PLANNED_COMPLEX: frozenset[str] = frozenset({
    "complex32",     # future
    "complex64", "complex128",
})

_PLANNED_PACKED_INT: frozenset[str] = frozenset({
    "int4",          # packed direct int4 — distinct from int4 quantization
})

_PLANNED_AMD_MX: frozenset[str] = frozenset({
    "mxfp8", "mxfp6", "mxfp4",
})

_PLANNED_TT_BLOCK: frozenset[str] = frozenset({
    "bfp8", "bfp4",
    "blockfp8", "blockfp4",
})

_PLANNED_GATED_DTYPES: frozenset[str] = (
    _PLANNED_UNSIGNED
    | _PLANNED_COMPLEX
    | _PLANNED_PACKED_INT
    | _PLANNED_AMD_MX
    | _PLANNED_TT_BLOCK
)


# ──────────────────────────────────────────────────────────────────────────
# Aliases — accepted at API boundaries, normalized before storage
# ──────────────────────────────────────────────────────────────────────────

_DTYPE_ALIASES: dict[str, str] = {
    # FP family
    "f64": "fp64", "float64": "fp64", "double": "fp64",
    "f32": "fp32", "float32": "fp32", "float": "fp32",
    "f16": "fp16", "float16": "fp16", "half": "fp16",
    "bfloat16": "bf16",
    # Integer family (MLIR-style + numpy-style)
    "i8": "int8",
    "i16": "int16",
    "i32": "int32",
    "i64": "int64",
    # Boolean
    "i1": "bool", "boolean": "bool",
}


# ──────────────────────────────────────────────────────────────────────────
# TF32 — not a storage dtype.  Captured here so canonicalize() can emit a
# precise error pointing the user at numeric_policy instead.
# ──────────────────────────────────────────────────────────────────────────

_TF32_NOT_A_DTYPE: frozenset[str] = frozenset({"tf32", "TF32"})


# ──────────────────────────────────────────────────────────────────────────
# Public helpers
# ──────────────────────────────────────────────────────────────────────────

def canonical_dtypes() -> frozenset[str]:
    """Return the canonical dtype name set (frozen)."""
    return _CANONICAL_DTYPES


def planned_gated_dtypes() -> frozenset[str]:
    """Return the planned/gated dtype name set (frozen).

    Members are *recognized* by the registry but **not** first-class today.
    Acceptance requires the consuming entry to declare
    ``metadata.dtype_status = "planned_gated"``.
    """
    return _PLANNED_GATED_DTYPES


def dtype_aliases() -> dict[str, str]:
    """Return a copy of the alias → canonical map."""
    return dict(_DTYPE_ALIASES)


def is_canonical_dtype(s: str) -> bool:
    """True if ``s`` is a canonical dtype spelling (i.e., the stored form)."""
    return s in _CANONICAL_DTYPES


def is_planned_gated_dtype(s: str) -> bool:
    """True if ``s`` is a recognized-but-gated dtype.

    Recognized-but-gated means the registry will accept the name when
    properly tagged, but no first-class storage path exists yet.
    """
    return s in _PLANNED_GATED_DTYPES


def is_known_dtype(s: str) -> bool:
    """True if ``s`` resolves to a canonical OR planned-gated dtype.

    Aliases (``"f32"`` → ``"fp32"``) count as known.  TF32 does NOT
    (TF32 is not a storage dtype — see ``canonicalize_dtype`` errors).
    """
    if s in _CANONICAL_DTYPES or s in _PLANNED_GATED_DTYPES:
        return True
    if s in _DTYPE_ALIASES:
        return True
    return False


def canonicalize_dtype(
    s: str,
    *,
    allow_planned_gated: bool = False,
) -> str:
    """Normalize a dtype string to its canonical spelling.

    Examples
    --------
    >>> canonicalize_dtype("f32")
    'fp32'
    >>> canonicalize_dtype("FP16")
    'fp16'
    >>> canonicalize_dtype("bf16")
    'bf16'
    >>> canonicalize_dtype("tf32")     # raises
    Traceback (most recent call last):
        ...
    tessera.dtype.TesseraDtypeError: ...

    Parameters
    ----------
    s : str
        Input spelling. Case-insensitive for the alias map; canonical
        spellings are case-sensitive (``"FP32"`` is rejected — use
        ``"fp32"``).
    allow_planned_gated : bool, default False
        If True, allow names from the planned/gated set
        (``uint*``/``complex*``/``int4``/``mxfp*``/``bfp*``).  The caller
        accepts responsibility for declaring ``dtype_status="planned_gated"``
        in any registry metadata.
    """
    if not isinstance(s, str):
        raise TesseraDtypeError(
            f"dtype must be a string, got {type(s).__name__}: {s!r}"
        )
    if not s:
        raise TesseraDtypeError("dtype string is empty")
    # Compound spellings (e.g., "bf16/fp32") belong on numeric_policy, not
    # on the storage dtype.  Reject early with a precise message.
    if "/" in s or "," in s or "+" in s:
        raise TesseraDtypeError(
            f"dtype {s!r} looks compound (storage/accumulator).  "
            "Accumulator + scaling belong on numeric_policy, not on the "
            "dtype string.  See docs/reference/tessera_tensor_attributes.md "
            "#numeric_policy."
        )
    # TF32 has a dedicated error so callers see exactly why it's rejected.
    if s in _TF32_NOT_A_DTYPE:
        raise TesseraDtypeError(
            "tf32 is not a storage dtype.  Model as math_mode='tf32' on an "
            "fp32 tensor via numeric_policy.  See "
            "docs/reference/tessera_tensor_attributes.md (Planned Or Gated "
            "Dtypes table)."
        )

    # Already canonical?
    if s in _CANONICAL_DTYPES:
        return s

    # Aliased?  We accept canonical-case aliases plus a single lowercase
    # fold so "FP32"/"Fp32" route through "fp32".
    if s in _DTYPE_ALIASES:
        return _DTYPE_ALIASES[s]
    lower = s.lower()
    if lower in _CANONICAL_DTYPES:
        return lower
    if lower in _DTYPE_ALIASES:
        return _DTYPE_ALIASES[lower]

    # Planned/gated?
    if s in _PLANNED_GATED_DTYPES or lower in _PLANNED_GATED_DTYPES:
        normalized = s if s in _PLANNED_GATED_DTYPES else lower
        if allow_planned_gated:
            return normalized
        raise TesseraDtypeError(
            f"dtype {s!r} is in the planned/gated set "
            f"({sorted(_PLANNED_GATED_DTYPES)}).  Pass "
            "allow_planned_gated=True AND declare "
            "metadata.dtype_status='planned_gated' on the consuming "
            "registry entry.  See docs/reference/tessera_tensor_attributes.md."
        )

    raise TesseraDtypeError(
        f"unknown dtype {s!r}.  Canonical dtypes: "
        f"{sorted(_CANONICAL_DTYPES)}.  See "
        "docs/reference/tessera_tensor_attributes.md for the full table "
        "(canonical + accepted aliases + planned/gated set)."
    )


def assert_canonical_dtype(
    s: str,
    *,
    allow_planned_gated: bool = False,
    context: str | None = None,
) -> str:
    """Assert ``s`` is canonical (or canonicalizable) and return the canonical form.

    Raises ``TesseraDtypeError`` with optional caller context attached.
    Useful at API boundaries that want a single line to enforce the rule.
    """
    try:
        return canonicalize_dtype(s, allow_planned_gated=allow_planned_gated)
    except TesseraDtypeError as e:
        if context:
            raise TesseraDtypeError(f"{context}: {e}") from None
        raise


__all__ = [
    "TesseraDtypeError",
    "canonicalize_dtype",
    "assert_canonical_dtype",
    "is_canonical_dtype",
    "is_planned_gated_dtype",
    "is_known_dtype",
    "canonical_dtypes",
    "planned_gated_dtypes",
    "dtype_aliases",
]
