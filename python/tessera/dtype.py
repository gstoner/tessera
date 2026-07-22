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
   ``complex64/128``, and AMD ``mxfp8/6/4`` are recognized by ``is_planned_gated_dtype``
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
    "int4", "int8", "int16", "int32", "int64",
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

_PLANNED_AMD_MX: frozenset[str] = frozenset({
    "mxfp8", "mxfp6", "mxfp4",
})

_PLANNED_GATED_DTYPES: frozenset[str] = (
    _PLANNED_UNSIGNED
    | _PLANNED_COMPLEX
    | _PLANNED_AMD_MX
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
    # AMD / rocWMMA instruction spelling.  Tessera stores the explicit
    # 8-bit float encodings; AMD docs often abbreviate E4M3 as FP8/F8
    # and E5M2-style bfloat8 as BF8.
    "f8": "fp8_e4m3", "fp8": "fp8_e4m3", "float8": "fp8_e4m3",
    "float8_e4m3fn": "fp8_e4m3",
    "bf8": "fp8_e5m2", "bfloat8": "fp8_e5m2",
    "float8_e5m2": "fp8_e5m2",
    # Integer family (MLIR-style + numpy-style)
    "i8": "int8",
    "i4": "int4",
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
        (``uint*``/``complex*``/``mxfp*``).  The caller
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


# ──────────────────────────────────────────────────────────────────────────
# Sprint F — public Dtype object + result_type lattice (2026-05-11)
#
# A typed wrapper around the canonical dtype string that's still
# `str`-compatible for back-compat with the existing string-based API.
# Lets callers write ``ts.dtype.Dtype("fp32")`` or pass a `Dtype` instance
# anywhere a string is accepted today.
#
# ``result_type(*dtypes, mode="standard"|"strict")`` returns the result
# of NumPy-style implicit promotion (``standard``) or rejects mixed
# dtypes (``strict``).  See ``docs/reference/tessera_tensor_attributes.md``
# JAX comparison section.
# ──────────────────────────────────────────────────────────────────────────


# Promotion ordering — wider (or strictly more expressive) wins.  The
# integer family promotes by bit width; the float family promotes to the
# wider mantissa+exponent.  bf16 and fp16 are deliberately incomparable
# (different exponent ranges) — mixing them promotes to fp32 for safety.
_FLOAT_ORDER: tuple[str, ...] = ("fp16", "bf16", "fp32", "fp64")
_INT_ORDER:   tuple[str, ...] = ("bool", "int4", "int8", "int16", "int32", "int64")
_LOW_PRECISION: frozenset[str] = _CANONICAL_LOW_PRECISION


def _is_float(s: str) -> bool:
    return s in _FLOAT_ORDER or s in _LOW_PRECISION


def _is_int(s: str) -> bool:
    return s in _INT_ORDER


def _promote_two(a: str, b: str) -> str:
    """NumPy-style implicit promotion of two canonical dtypes.

    Rules (matching the doc's "Promotion And Casting Policy"):
      1. Identical → unchanged.
      2. Low-precision (fp8 / fp6 / fp4 / nvfp4) is storage-only;
         arithmetic with anything else promotes to fp32 (or the wider
         partner's family if the other operand is fp64).
      3. Float + float: wider wins.  bf16 + fp16 → fp32 (incomparable
         exponent ranges; choose the safe upper bound).
      4. Int + int: wider wins (bool < int8 < int16 < int32 < int64).
      5. Float + int: float wins.  fp16/bf16 + int32 → fp32 to avoid
         silent precision loss.
    """
    if a == b:
        return a

    a_low = a in _LOW_PRECISION
    b_low = b in _LOW_PRECISION

    if a_low and b_low:
        return "fp32"
    if a_low:
        return _promote_two("fp32", b)
    if b_low:
        return _promote_two(a, "fp32")

    a_f = _is_float(a)
    b_f = _is_float(b)
    a_i = _is_int(a)
    b_i = _is_int(b)

    # float + int → float side, but never narrower than fp32 when the int
    # is wider than int16
    if a_f and b_i:
        if b in ("int32", "int64") and a in ("fp16", "bf16"):
            return "fp32"
        return a
    if a_i and b_f:
        if a in ("int32", "int64") and b in ("fp16", "bf16"):
            return "fp32"
        return b

    if a_f and b_f:
        # bf16 + fp16 → fp32 (incomparable exponent ranges)
        if {a, b} == {"bf16", "fp16"}:
            return "fp32"
        idx_a = _FLOAT_ORDER.index(a)
        idx_b = _FLOAT_ORDER.index(b)
        return _FLOAT_ORDER[max(idx_a, idx_b)]

    if a_i and b_i:
        idx_a = _INT_ORDER.index(a)
        idx_b = _INT_ORDER.index(b)
        return _INT_ORDER[max(idx_a, idx_b)]

    raise TesseraDtypeError(
        f"no promotion rule for ({a!r}, {b!r}); both must be canonical "
        "float or integer dtypes."
    )


def result_type(*dtypes, mode: str = "standard") -> "Dtype":
    """Compute the promotion result of one or more dtypes.

    Parameters
    ----------
    *dtypes : str | Dtype
        Two or more dtype specs; each is canonicalized before promotion.
    mode : str, default "standard"
        ``"standard"`` — NumPy/JAX-style implicit promotion.
        ``"strict"``    — reject any mixed-dtype combination unless every
                          argument resolves to the same canonical dtype.
                          Matches JAX's ``jax_numpy_dtype_promotion='strict'``.

    Returns
    -------
    Dtype
        The promoted canonical dtype.  Always a ``Dtype`` instance
        (str-compatible).

    Raises
    ------
    TesseraDtypeError
        On strict mode with mixed dtypes, planned-gated input, or
        unknown spellings.
    """
    if not dtypes:
        raise TesseraDtypeError("result_type requires at least one dtype")
    if mode not in {"standard", "strict"}:
        raise TesseraDtypeError(
            f"result_type mode must be 'standard' or 'strict', got {mode!r}"
        )
    canonical = [canonicalize_dtype(str(d)) for d in dtypes]
    if mode == "strict":
        unique = set(canonical)
        if len(unique) > 1:
            raise TesseraDtypeError(
                f"strict-mode result_type rejects mixed dtypes {sorted(unique)}.  "
                "Insert an explicit cast or switch to mode='standard'."
            )
        return Dtype(canonical[0])

    # Standard mode — left-fold _promote_two over the canonical list.
    out = canonical[0]
    for d in canonical[1:]:
        out = _promote_two(out, d)
    return Dtype(out)


class Dtype(str):
    """Str-compatible typed wrapper around a canonical dtype name.

    Created via:
        Dtype("fp32"), Dtype("f32") → "fp32"  (alias normalized)
        Dtype.from_(x) where x is already a Dtype → same object
        Dtype.canonical("fp16")    → Dtype instance

    Subclasses ``str`` so existing code that does ``arr.dtype == "fp32"``,
    ``np.dtype(s)``, ``f"{x.dtype}"`` keeps working unchanged.  The class
    adds a small object-oriented surface on top.

    Use cases:
      - ``ts.dtype.Dtype("f32") == "fp32"``  → True (canonicalization)
      - ``ts.dtype.Dtype("bf16").is_float``  → True
      - ``ts.dtype.Dtype("fp32").bits``      → 32
      - ``ts.dtype.Dtype("fp16") | ts.dtype.Dtype("bf16")`` → Dtype("fp32")
        (operator-style promotion via ``__or__`` — same rule as
        ``result_type(a, b, mode='standard')``)
    """

    __slots__ = ()

    def __new__(cls, value, *, allow_planned_gated: bool = False):
        if isinstance(value, Dtype):
            return value  # idempotent
        if isinstance(value, str):
            canon = canonicalize_dtype(value, allow_planned_gated=allow_planned_gated)
        else:
            raise TesseraDtypeError(
                f"Dtype expects a string-like value, got {type(value).__name__}"
            )
        return str.__new__(cls, canon)

    @classmethod
    def canonical(cls, value: str) -> "Dtype":
        """Construct a Dtype from a canonical or aliased string."""
        return cls(value)

    @classmethod
    def from_(cls, value) -> "Dtype":
        """Lift a string-or-Dtype to Dtype, idempotent on Dtype inputs."""
        return value if isinstance(value, Dtype) else cls(value)

    @property
    def is_canonical(self) -> bool:
        return is_canonical_dtype(str(self))

    @property
    def is_planned_gated(self) -> bool:
        return is_planned_gated_dtype(str(self))

    @property
    def is_float(self) -> bool:
        return _is_float(str(self))

    @property
    def is_integer(self) -> bool:
        return _is_int(str(self))

    @property
    def is_low_precision(self) -> bool:
        return str(self) in _LOW_PRECISION

    @property
    def bits(self) -> int:
        """Storage width in bits.  Block-scaled formats report their nominal
        element width (e.g., nvfp4 → 4)."""
        s = str(self)
        if s == "bool":
            return 1
        if s.startswith("int"):
            return int(s[3:])
        if s == "fp64":
            return 64
        if s == "fp32":
            return 32
        if s in ("fp16", "bf16"):
            return 16
        if s.startswith("fp8"):
            return 8
        if s.startswith("fp6"):
            return 6
        if s.startswith("fp4") or s == "nvfp4":
            return 4
        raise TesseraDtypeError(f"unknown bit width for dtype {s!r}")

    def __or__(self, other) -> "Dtype":
        """Operator-style promotion: ``Dtype('fp16') | Dtype('bf16')`` →
        ``Dtype('fp32')``.  Equivalent to
        ``result_type(self, other, mode='standard')``."""
        if not isinstance(other, (str, Dtype)):
            return NotImplemented
        return result_type(self, other, mode="standard")

    __ror__ = __or__

    def __repr__(self) -> str:
        return f"Dtype({str(self)!r})"


# Short aliases — the doc's "JAX comparison" section names these as the
# expected public surface.  Both spellings stay live; the long names
# (``canonicalize_dtype`` / ``is_canonical_dtype``) document the canonical
# vs. alias distinction, while the short names (``canonicalize`` /
# ``is_canonical``) match the JAX-side vocabulary the doc references.
canonicalize = canonicalize_dtype
is_canonical = is_canonical_dtype
is_planned_gated = is_planned_gated_dtype


__all__ = [
    "TesseraDtypeError",
    "Dtype",
    "result_type",
    "canonicalize_dtype",
    "assert_canonical_dtype",
    "is_canonical_dtype",
    "is_planned_gated_dtype",
    "is_known_dtype",
    "canonical_dtypes",
    "planned_gated_dtypes",
    "dtype_aliases",
    # Short aliases (doc-referenced JAX-style spelling)
    "canonicalize",
    "is_canonical",
    "is_planned_gated",
]
