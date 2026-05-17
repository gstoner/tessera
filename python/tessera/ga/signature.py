"""GA1 — Clifford algebra signature object.

A `Cl(p, q, r=0)` is a hashable, constructable signature with a
compile-time-computed geometric-product table. The signature object is
the configuration root for everything downstream in the GA-series:
GA2 grade-aware types, GA3 multivectors, GA4 primitive_coverage entries,
GA7 the `tessera.clifford` dialect, GA8/GA9 lowering and backends.

V1 allow-list (per docs/audit/ga_scope_lock.md):

    Cl(3, 0, 0) — 3D Euclidean. 8 basis elements.
    Cl(1, 3, 0) — Minkowski spacetime. 16 basis elements.

Other signatures raise `TesseraAlgebraError` at construction. The object
is parameterized for general `(p, q, r)` so v1 → vN is a guard-relaxation,
not a redesign.

Basis convention. Generators are 1-indexed (`e1, e2, ...`). A blade is
identified by a bitmask: bit `i` set means generator `e_{i+1}` is in the
blade's wedge product, with generators in ascending order. Bitmask 0 is
the scalar (canonical name "1"); bitmask `(1 << n) - 1` is the
pseudoscalar.

Geometric product. For two blades `A` and `B` with bitmasks `mask_a` and
`mask_b`, the product is `±C` where:
  - the result bitmask is `mask_a XOR mask_b` (symmetric difference)
  - the sign comes from reordering swaps PLUS the signature of each
    squared generator (`e_i^2 = +1` for first `p`, `-1` for next `q`,
    `0` for next `r` — the latter zeroes the product entirely).

All signature-derived data (basis list, product table) is computed once
per signature and cached via `functools.lru_cache`.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Tuple


# v1 allow-list per Q1 in docs/audit/ga_scope_lock.md.
V1_ALLOWED_SIGNATURES: frozenset[Tuple[int, int, int]] = frozenset(
    {
        (3, 0, 0),  # 3D Euclidean
        (1, 3, 0),  # Minkowski spacetime
    }
)


class TesseraAlgebraError(ValueError):
    """Raised on invalid Clifford signature or basis access."""


@dataclass(frozen=True)
class Basis:
    """A canonical basis blade of a Clifford algebra.

    Fields:
        mask: bitmask in `[0, 2**n)` identifying which generators are
            wedged into this blade. Bit `i` set ⇒ generator `e_{i+1}`
            is present.
        name: canonical string form. Scalar is `"1"`; vector `"e1"`;
            bivector `"e12"`; pseudoscalar of Cl(3,0) is `"e123"`.
        grade: number of generators in the blade (popcount of `mask`).
    """

    mask: int
    name: str
    grade: int


def _basis_name(mask: int) -> str:
    if mask == 0:
        return "1"
    indices: list[str] = []
    i = 0
    m = mask
    while m:
        if m & 1:
            indices.append(str(i + 1))
        m >>= 1
        i += 1
    return "e" + "".join(indices)


def _name_to_mask(name: str, n: int) -> int:
    if name == "1":
        return 0
    if not name.startswith("e") or len(name) < 2:
        raise TesseraAlgebraError(
            f"Invalid blade name {name!r}; expected '1' or 'e<indices>' "
            f"(e.g. 'e1', 'e12')."
        )
    digits = name[1:]
    if not digits.isdigit():
        raise TesseraAlgebraError(
            f"Invalid blade name {name!r}; indices must be digits."
        )
    seen: set[int] = set()
    last = 0
    mask = 0
    for ch in digits:
        idx = int(ch)
        if idx < 1 or idx > n:
            raise TesseraAlgebraError(
                f"Blade {name!r} references generator e{idx} but algebra has "
                f"only {n} generators."
            )
        if idx in seen:
            raise TesseraAlgebraError(
                f"Blade {name!r} repeats generator e{idx}; canonical blade "
                f"names use each generator at most once."
            )
        if idx <= last:
            raise TesseraAlgebraError(
                f"Blade {name!r} is not in canonical (ascending) order. "
                f"Use e.g. 'e12' instead of 'e21' for the bivector; sign "
                f"flips come from the product table, not the name."
            )
        seen.add(idx)
        last = idx
        mask |= 1 << (idx - 1)
    return mask


def _blade_product(
    mask_a: int, mask_b: int, p: int, q: int, r: int
) -> Tuple[int, int]:
    """Compute the geometric product of two basis blades.

    Returns ``(result_mask, sign)``. If the result is zero (any null
    generator squared), returns ``(0, 0)``.
    """
    n = p + q + r
    # Reordering sign: for each generator e_i in B (bit i in mask_b), count
    # how many generators e_j with j > i are present in A. Each such pair
    # is one swap to slide e_i past e_j.
    sign = 1
    for i in range(n):
        if (mask_b >> i) & 1:
            higher_a = mask_a >> (i + 1)
            # Count set bits in higher_a.
            if higher_a.bit_count() & 1:
                sign = -sign
    # Symmetric difference: generators present in exactly one of A, B
    # remain in the product. Generators in both square and disappear,
    # contributing their signature value.
    common = mask_a & mask_b
    result_mask = mask_a ^ mask_b
    if common:
        # First p generators square to +1 (no sign change).
        # Next q generators square to -1.
        # Last r generators square to 0 — the whole product is zero.
        q_mask = ((1 << (p + q)) - 1) & ~((1 << p) - 1)
        r_mask = ((1 << n) - 1) & ~((1 << (p + q)) - 1)
        if common & r_mask:
            return 0, 0
        neg_count = (common & q_mask).bit_count()
        if neg_count & 1:
            sign = -sign
    return result_mask, sign


@lru_cache(maxsize=None)
def _product_table(
    p: int, q: int, r: int
) -> Tuple[Tuple[Tuple[int, int], ...], ...]:
    """Full geometric-product Cayley table as (result_mask, sign) pairs.

    Indexed `table[a][b] == (result_mask, sign)` where `a`, `b` are
    blade bitmasks. Cached per signature.
    """
    dim = 1 << (p + q + r)
    return tuple(
        tuple(_blade_product(a, b, p, q, r) for b in range(dim))
        for a in range(dim)
    )


@lru_cache(maxsize=None)
def _basis_list(p: int, q: int, r: int) -> Tuple[Basis, ...]:
    n = p + q + r
    dim = 1 << n
    return tuple(
        Basis(mask=m, name=_basis_name(m), grade=m.bit_count())
        for m in range(dim)
    )


@dataclass(frozen=True)
class Cl:
    """A Clifford algebra signature ``Cl(p, q, r)``.

    ``p`` = number of generators that square to +1 (Euclidean).
    ``q`` = number of generators that square to −1 (Minkowski-like).
    ``r`` = number of generators that square to 0 (null/degenerate).

    Instances are hashable and value-equal: ``Cl(3, 0) == Cl(3, 0)``.
    Construction validates against the v1 allow-list (Cl(3,0) and Cl(1,3)
    only); other signatures raise `TesseraAlgebraError`. The validation
    can be lifted without touching downstream code; see
    ``docs/audit/ga_scope_lock.md`` § Q1 extension path.
    """

    p: int
    q: int
    r: int = 0

    def __post_init__(self) -> None:
        if not (isinstance(self.p, int) and isinstance(self.q, int) and isinstance(self.r, int)):
            raise TesseraAlgebraError(
                f"Cl({self.p}, {self.q}, {self.r}): all of p/q/r must be int."
            )
        if self.p < 0 or self.q < 0 or self.r < 0:
            raise TesseraAlgebraError(
                f"Cl({self.p}, {self.q}, {self.r}): all of p/q/r must be non-negative."
            )
        if self.signature not in V1_ALLOWED_SIGNATURES:
            raise TesseraAlgebraError(
                f"Cl{self.signature} is not in the v1 allow-list. "
                f"v1 supports {sorted(V1_ALLOWED_SIGNATURES)} only. "
                f"See docs/audit/ga_scope_lock.md § Q1."
            )

    @property
    def signature(self) -> Tuple[int, int, int]:
        """The signature triple ``(p, q, r)``."""
        return (self.p, self.q, self.r)

    @property
    def n(self) -> int:
        """Number of generators."""
        return self.p + self.q + self.r

    @property
    def dim(self) -> int:
        """Dimension of the algebra as a vector space: ``2**n``."""
        return 1 << self.n

    @property
    def grades(self) -> Tuple[int, ...]:
        """Tuple of valid grades ``(0, 1, ..., n)``."""
        return tuple(range(self.n + 1))

    @property
    def scalar(self) -> Basis:
        """The scalar blade (grade 0)."""
        return Basis(mask=0, name="1", grade=0)

    @property
    def pseudoscalar(self) -> Basis:
        """The top-grade blade (grade n) — `e_1 ∧ ... ∧ e_n`."""
        mask = self.dim - 1
        return Basis(mask=mask, name=_basis_name(mask), grade=self.n)

    def blade(self, name: str) -> Basis:
        """Look up a basis blade by canonical name.

        Accepts ``"1"`` (scalar) or ``"e<i1><i2>...<ik>"`` with
        ``i1 < i2 < ... < ik`` and each index in ``[1, n]``.
        """
        mask = _name_to_mask(name, self.n)
        return Basis(mask=mask, name=_basis_name(mask), grade=mask.bit_count())

    def blades(self) -> Tuple[Basis, ...]:
        """All ``2**n`` basis blades, in ascending bitmask order."""
        return _basis_list(self.p, self.q, self.r)

    def product_table(self) -> Tuple[Tuple[Tuple[int, int], ...], ...]:
        """The full Cayley table; ``table[a][b] == (mask, sign)``."""
        return _product_table(self.p, self.q, self.r)

    def blades_of_grade(self, grade: int) -> Tuple[Basis, ...]:
        """All basis blades with the given grade."""
        if grade < 0 or grade > self.n:
            raise TesseraAlgebraError(
                f"grade {grade} is out of range for Cl{self.signature}; "
                f"valid grades are {self.grades}."
            )
        return tuple(b for b in self.blades() if b.grade == grade)

    def __repr__(self) -> str:
        if self.r == 0:
            return f"Cl({self.p}, {self.q})"
        return f"Cl({self.p}, {self.q}, {self.r})"


__all__ = [
    "Basis",
    "Cl",
    "TesseraAlgebraError",
    "V1_ALLOWED_SIGNATURES",
]
