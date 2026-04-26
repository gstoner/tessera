"""
tessera.compiler.constraints — ConstraintSolver and predicate types.

Constraints are checked at @jit decoration time (not call time). They are
structural properties of the function's type signature, validated before
any IR is emitted.

Predicates:
    Divisible("K", 64)        — K % 64 == 0
    Range("S", 1, 8192)       — 1 <= S <= 8192
    Equal("D_in", "D_out")    — D_in == D_out

Usage:
    @tessera.jit
    def aligned_gemm(A: tessera.Tensor["M", "K"], B: tessera.Tensor["K", "N"]):
        tessera.require(tessera.constraint.Divisible("K", 64))
        return tessera.ops.gemm(A, B)

Violation raises TesseraConstraintError with the offending dimension path.

Reference: CLAUDE.md §Key Design Contracts — ConstraintSolver
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union


# ─────────────────────────────────────────────────────────────────────────────
# Error type
# ─────────────────────────────────────────────────────────────────────────────

class TesseraConstraintError(Exception):
    """
    Raised when a structural constraint is violated at @jit decoration time.

    Attributes:
        constraint : the predicate that failed
        dim_name   : the symbolic dimension name
        actual     : actual value (if available at decoration time)
        message    : human-readable explanation
    """

    def __init__(
        self,
        constraint: "Constraint",
        dim_name: str,
        actual: Optional[Any] = None,
        message: Optional[str] = None,
    ) -> None:
        self.constraint = constraint
        self.dim_name = dim_name
        self.actual = actual
        self._message = message
        super().__init__(self._build_message())

    def _build_message(self) -> str:
        if self._message:
            return self._message
        base = f"Constraint {self.constraint!r} violated for dimension {self.dim_name!r}"
        if self.actual is not None:
            base += f" (actual value: {self.actual})"
        return base


# ─────────────────────────────────────────────────────────────────────────────
# Predicate base class
# ─────────────────────────────────────────────────────────────────────────────

class Constraint:
    """
    Base class for all structural constraints.

    A Constraint is a predicate over one or more symbolic dimension names.
    It is collected by tessera.require() inside @jit bodies and checked by
    ConstraintSolver at decoration time.
    """

    def check(self, bindings: Dict[str, int]) -> Optional[TesseraConstraintError]:
        """
        Check this constraint against a concrete binding of dimension names → sizes.

        Args:
            bindings: dict mapping dimension name to concrete int size.
                      Dimensions not present in bindings are skipped (symbolic).

        Returns:
            None if constraint is satisfied or cannot be checked yet (symbolic dims),
            TesseraConstraintError if the constraint is violated.
        """
        raise NotImplementedError

    def dim_names(self) -> List[str]:
        """Return the symbolic dimension names this constraint references."""
        raise NotImplementedError


# ─────────────────────────────────────────────────────────────────────────────
# Predicate implementations
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Divisible(Constraint):
    """
    Assert that a dimension size is evenly divisible by a divisor.

    Catches misalignment early — e.g., K must be a multiple of 64 for
    tensor core GEMM to operate without padding.

    Args:
        dim   : symbolic dimension name (e.g., "K")
        divisor: required divisor (positive int)

    Example:
        tessera.require(Divisible("K", 64))
        # → raises TesseraConstraintError if K % 64 != 0
    """
    dim: str
    divisor: int

    def __post_init__(self) -> None:
        if not isinstance(self.divisor, int) or self.divisor < 1:
            raise ValueError(
                f"Divisible.divisor must be a positive int, got {self.divisor!r}"
            )
        if not isinstance(self.dim, str) or not self.dim:
            raise ValueError(
                f"Divisible.dim must be a non-empty string, got {self.dim!r}"
            )

    def check(self, bindings: Dict[str, int]) -> Optional[TesseraConstraintError]:
        if self.dim not in bindings:
            return None  # symbolic — cannot check yet
        val = bindings[self.dim]
        if val % self.divisor != 0:
            return TesseraConstraintError(
                constraint=self,
                dim_name=self.dim,
                actual=val,
                message=(
                    f"Dimension {self.dim!r} = {val} is not divisible by {self.divisor}. "
                    f"Required: {self.dim} % {self.divisor} == 0"
                ),
            )
        return None

    def dim_names(self) -> List[str]:
        return [self.dim]

    def __repr__(self) -> str:
        return f"Divisible({self.dim!r}, {self.divisor})"


@dataclass(frozen=True)
class Range(Constraint):
    """
    Assert that a dimension size falls within [lo, hi] inclusive.

    Useful for bounding sequence lengths, preventing OOM for known-bad
    shapes, or catching underflow (e.g. S >= 1).

    Args:
        dim : symbolic dimension name (e.g., "S")
        lo  : minimum allowed value (inclusive)
        hi  : maximum allowed value (inclusive)

    Example:
        tessera.require(Range("S", 1, 8192))
        # → raises TesseraConstraintError if S < 1 or S > 8192
    """
    dim: str
    lo: int
    hi: int

    def __post_init__(self) -> None:
        if not isinstance(self.lo, int) or not isinstance(self.hi, int):
            raise ValueError(f"Range bounds must be ints, got lo={self.lo!r}, hi={self.hi!r}")
        if self.lo > self.hi:
            raise ValueError(
                f"Range lo must be <= hi, got lo={self.lo}, hi={self.hi}"
            )
        if not isinstance(self.dim, str) or not self.dim:
            raise ValueError(f"Range.dim must be a non-empty string, got {self.dim!r}")

    def check(self, bindings: Dict[str, int]) -> Optional[TesseraConstraintError]:
        if self.dim not in bindings:
            return None
        val = bindings[self.dim]
        if not (self.lo <= val <= self.hi):
            return TesseraConstraintError(
                constraint=self,
                dim_name=self.dim,
                actual=val,
                message=(
                    f"Dimension {self.dim!r} = {val} is out of range "
                    f"[{self.lo}, {self.hi}]. Required: {self.lo} <= {self.dim} <= {self.hi}"
                ),
            )
        return None

    def dim_names(self) -> List[str]:
        return [self.dim]

    def __repr__(self) -> str:
        return f"Range({self.dim!r}, {self.lo}, {self.hi})"


@dataclass(frozen=True)
class Equal(Constraint):
    """
    Assert that two dimension sizes are equal.

    Catches shape mismatches early — e.g., the inner dimension of A must
    equal the outer dimension of B in a GEMM.

    Args:
        dim_a : first symbolic dimension name (e.g., "D_in")
        dim_b : second symbolic dimension name (e.g., "D_out")

    Example:
        tessera.require(Equal("D_in", "D_out"))
        # → raises TesseraConstraintError if D_in != D_out
    """
    dim_a: str
    dim_b: str

    def __post_init__(self) -> None:
        for name, val in [("dim_a", self.dim_a), ("dim_b", self.dim_b)]:
            if not isinstance(val, str) or not val:
                raise ValueError(f"Equal.{name} must be a non-empty string, got {val!r}")

    def check(self, bindings: Dict[str, int]) -> Optional[TesseraConstraintError]:
        if self.dim_a not in bindings or self.dim_b not in bindings:
            return None  # one or both symbolic — skip
        a = bindings[self.dim_a]
        b = bindings[self.dim_b]
        if a != b:
            return TesseraConstraintError(
                constraint=self,
                dim_name=f"{self.dim_a}/{self.dim_b}",
                actual=(a, b),
                message=(
                    f"Dimension equality constraint violated: "
                    f"{self.dim_a!r} = {a} != {self.dim_b!r} = {b}"
                ),
            )
        return None

    def dim_names(self) -> List[str]:
        return [self.dim_a, self.dim_b]

    def __repr__(self) -> str:
        return f"Equal({self.dim_a!r}, {self.dim_b!r})"


# ─────────────────────────────────────────────────────────────────────────────
# ConstraintSolver
# ─────────────────────────────────────────────────────────────────────────────

class ConstraintSolver:
    """
    Collects and checks structural constraints on a @jit-decorated function.

    ConstraintSolver.check() is called by @jit at decoration time. It walks
    the function's annotation signature to extract symbolic dimension names,
    then evaluates each registered constraint against any concrete bindings.

    Symbolic dimensions (unknown at decoration time) are skipped. Concrete
    violations raise TesseraConstraintError immediately.

    Usage:
        solver = ConstraintSolver()
        solver.add(Divisible("K", 64))
        solver.add(Range("S", 1, 8192))
        solver.check(bindings={"K": 128, "S": 4096})   # passes
        solver.check(bindings={"K": 100})               # raises: 100 % 64 != 0

    The solver accumulates constraints across multiple add() calls and checks
    all of them on each check() invocation.
    """

    def __init__(self) -> None:
        self._constraints: List[Constraint] = []

    def add(self, constraint: Constraint) -> "ConstraintSolver":
        """Register a constraint. Returns self for chaining."""
        if not isinstance(constraint, Constraint):
            raise TypeError(
                f"Expected a Constraint instance, got {type(constraint).__name__!r}"
            )
        self._constraints.append(constraint)
        return self

    def check(self, bindings: Optional[Dict[str, int]] = None) -> None:
        """
        Check all registered constraints against the given bindings.

        Args:
            bindings: dict of dim_name → concrete int size. Symbolic dims
                      (not in bindings) are skipped.

        Raises:
            TesseraConstraintError: on the first violated constraint.
        """
        if bindings is None:
            bindings = {}
        for constraint in self._constraints:
            error = constraint.check(bindings)
            if error is not None:
                raise error

    def check_all(self, bindings: Optional[Dict[str, int]] = None) -> List[TesseraConstraintError]:
        """
        Check all constraints and return ALL violations (not just the first).

        Useful for diagnostic reporting where you want to surface all problems
        at once rather than stopping at the first error.

        Returns:
            List of TesseraConstraintError for each violated constraint.
        """
        if bindings is None:
            bindings = {}
        errors = []
        for constraint in self._constraints:
            error = constraint.check(bindings)
            if error is not None:
                errors.append(error)
        return errors

    @property
    def constraints(self) -> List[Constraint]:
        """Read-only view of registered constraints."""
        return list(self._constraints)

    def __len__(self) -> int:
        return len(self._constraints)

    def __repr__(self) -> str:
        return f"ConstraintSolver({self._constraints!r})"
