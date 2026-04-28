"""Shape-system foundation for Tessera.

This module is intentionally lightweight: it gives the Python frontend and
tests a shared representation for symbolic dimensions, derived dimensions,
broadcasting, sharding checks, schedule feasibility, and runtime witnesses.
The compiler-facing MLIR verifier remains the source of truth for final
lowering legality.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import reduce
from operator import mul
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence, Union


DimLike = Union[int, "Dim", "DimProduct"]


class ShapeSystemError(ValueError):
    """Raised when shape, layout, shard, or schedule constraints are invalid."""


@dataclass(frozen=True)
class Dim:
    """A symbolic or concrete tensor dimension."""

    name: str
    value: Optional[int] = None

    def __post_init__(self) -> None:
        if not self.name or any(ch.isspace() for ch in self.name):
            raise ValueError("dimension names must be non-empty and contain no whitespace")
        if self.value is not None and self.value <= 0:
            raise ValueError("concrete dimensions must be positive")

    def __mul__(self, other: DimLike) -> "DimProduct":
        return DimProduct((self, other))

    def __rmul__(self, other: DimLike) -> "DimProduct":
        return DimProduct((other, self))

    def eval(self, bindings: Mapping[str, int]) -> Optional[int]:
        if self.value is not None:
            return self.value
        return bindings.get(self.name)

    def __str__(self) -> str:
        return f"{self.name}={self.value}" if self.value is not None else self.name


@dataclass(frozen=True)
class DimProduct:
    """A product expression such as ``H * Dh`` used for derived dimensions."""

    factors: tuple[DimLike, ...]

    def __post_init__(self) -> None:
        flat: list[DimLike] = []
        for factor in self.factors:
            if isinstance(factor, DimProduct):
                flat.extend(factor.factors)
            else:
                flat.append(factor)
        object.__setattr__(self, "factors", tuple(flat))

    def __mul__(self, other: DimLike) -> "DimProduct":
        return DimProduct((*self.factors, other))

    def __rmul__(self, other: DimLike) -> "DimProduct":
        return DimProduct((other, *self.factors))

    def eval(self, bindings: Mapping[str, int]) -> Optional[int]:
        values: list[int] = []
        for factor in self.factors:
            value = _eval_dim(factor, bindings)
            if value is None:
                return None
            values.append(value)
        return reduce(mul, values, 1)

    def __str__(self) -> str:
        return " * ".join(_dim_label(f) for f in self.factors)


@dataclass(frozen=True)
class Layout:
    """Logical or physical tensor layout contract."""

    name: str
    params: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("layout name must be non-empty")


@dataclass(frozen=True)
class Shape:
    """A tensor shape with optional layout and sharding metadata."""

    dims: tuple[DimLike, ...]
    layout: Optional[Layout] = None
    shard: Optional["ShapeShard"] = None

    def __iter__(self):
        return iter(self.dims)

    def __len__(self) -> int:
        return len(self.dims)

    def __getitem__(self, index):
        return self.dims[index]

    def with_layout(self, layout: Layout) -> "Shape":
        return Shape(self.dims, layout=layout, shard=self.shard)

    def with_shard(self, shard: "ShapeShard") -> "Shape":
        return Shape(self.dims, layout=self.layout, shard=shard)


@dataclass(frozen=True)
class ShapeShard:
    """Maps logical dimensions to mesh axes, e.g. ``{"B": "dp", "D": "tp"}``."""

    dim_to_axis: Mapping[Union[str, int], str]


@dataclass(frozen=True)
class ShapeDiagnostic:
    """A human-readable shape-system violation."""

    code: str
    message: str
    suggestion: str = ""


@dataclass(frozen=True)
class ScheduleFeasibility:
    """Result of checking a schedule knob against a shape dimension."""

    ok: bool
    dim: DimLike
    tile: int
    actual: Optional[int] = None
    padded: Optional[int] = None

    @property
    def suggestion(self) -> str:
        if self.ok or self.actual is None or self.padded is None:
            return ""
        return f"pad {_dim_label(self.dim)} from {self.actual} to {self.padded}"


class ShapeConstraintGraph:
    """Small affine/integer constraint graph for Python-level shape checking."""

    def __init__(self) -> None:
        self._constraints: list[Callable[[Mapping[str, int]], Optional[ShapeDiagnostic]]] = []

    def equal(self, lhs: DimLike, rhs: DimLike) -> "ShapeConstraintGraph":
        def check(bindings: Mapping[str, int]) -> Optional[ShapeDiagnostic]:
            lv = _eval_dim(lhs, bindings)
            rv = _eval_dim(rhs, bindings)
            if lv is None or rv is None or lv == rv:
                return None
            return ShapeDiagnostic(
                "shape-equal",
                f"dimension equality failed: {_dim_label(lhs)}={lv} != {_dim_label(rhs)}={rv}",
                f"make {_dim_label(lhs)} and {_dim_label(rhs)} equal",
            )

        self._constraints.append(check)
        return self

    def derived(self, target: DimLike, expr: DimLike) -> "ShapeConstraintGraph":
        return self.equal(target, expr)

    def divisible(self, dim: DimLike, divisor: int) -> "ShapeConstraintGraph":
        if divisor <= 0:
            raise ValueError("divisor must be positive")

        def check(bindings: Mapping[str, int]) -> Optional[ShapeDiagnostic]:
            value = _eval_dim(dim, bindings)
            if value is None or value % divisor == 0:
                return None
            padded = next_multiple(value, divisor)
            return ShapeDiagnostic(
                "shape-divisible",
                f"{_dim_label(dim)}={value} is not divisible by {divisor}",
                f"pad {_dim_label(dim)} to {padded} or choose a compatible tile/mesh axis",
            )

        self._constraints.append(check)
        return self

    def range(self, dim: DimLike, lo: int, hi: int) -> "ShapeConstraintGraph":
        if lo > hi:
            raise ValueError("range lower bound must be <= upper bound")

        def check(bindings: Mapping[str, int]) -> Optional[ShapeDiagnostic]:
            value = _eval_dim(dim, bindings)
            if value is None or lo <= value <= hi:
                return None
            return ShapeDiagnostic(
                "shape-range",
                f"{_dim_label(dim)}={value} is outside [{lo}, {hi}]",
                f"constrain {_dim_label(dim)} to [{lo}, {hi}]",
            )

        self._constraints.append(check)
        return self

    def check_all(self, bindings: Mapping[str, int]) -> list[ShapeDiagnostic]:
        return [d for c in self._constraints if (d := c(bindings)) is not None]

    def raise_if_errors(self, bindings: Mapping[str, int]) -> None:
        errors = self.check_all(bindings)
        if errors:
            first = errors[0]
            suffix = f" suggestion: {first.suggestion}" if first.suggestion else ""
            raise ShapeSystemError(f"{first.code}: {first.message}{suffix}")


@dataclass(frozen=True)
class RuntimeShapeWitness:
    """Runtime assertion payload for dynamic dimensions."""

    name: str
    shape: tuple[DimLike, ...]
    constraints: ShapeConstraintGraph = field(default_factory=ShapeConstraintGraph)

    def refine(self, bindings: Mapping[str, int]) -> tuple[int, ...]:
        self.constraints.raise_if_errors(bindings)
        concrete: list[int] = []
        for dim in self.shape:
            value = _eval_dim(dim, bindings)
            if value is None:
                raise ShapeSystemError(f"missing runtime witness for {_dim_label(dim)}")
            concrete.append(value)
        return tuple(concrete)


def sym(names: str) -> Union[Dim, tuple[Dim, ...]]:
    """Create one or more symbolic dimensions.

    ``B, N, D = tessera.sym("B N D")`` is the intended user-facing form.
    """

    dims = tuple(Dim(name) for name in names.split())
    if not dims:
        raise ValueError("sym() requires at least one dimension name")
    return dims[0] if len(dims) == 1 else dims


def dim(name: str, value: Optional[int] = None) -> Dim:
    return Dim(name, value)


def broadcast_shape(*shapes: Sequence[DimLike]) -> tuple[DimLike, ...]:
    """Return the NumPy-style broadcast result for symbolic/concrete shapes."""

    if not shapes:
        return ()
    rev_shapes = [tuple(s)[::-1] for s in shapes]
    out: list[DimLike] = []
    for axis in range(max(len(s) for s in rev_shapes)):
        candidates = [s[axis] for s in rev_shapes if axis < len(s)]
        chosen = candidates[0]
        for candidate in candidates[1:]:
            chosen = _broadcast_dim(chosen, candidate)
        out.append(chosen)
    return tuple(out[::-1])


def matmul_shape(lhs: Sequence[DimLike], rhs: Sequence[DimLike]) -> tuple[DimLike, ...]:
    L, R = tuple(lhs), tuple(rhs)
    if len(L) < 2 or len(R) < 2:
        raise ShapeSystemError(f"matmul requires rank >= 2; got {L} x {R}")
    if not dims_compatible(L[-1], R[-2]):
        raise ShapeSystemError(
            f"matmul inner dimensions differ: {_dim_label(L[-1])} vs {_dim_label(R[-2])}"
        )
    return broadcast_shape(L[:-2], R[:-2]) + (L[-2], R[-1])


def reshape_shape(
    source: Sequence[DimLike],
    target: Sequence[DimLike],
    bindings: Optional[Mapping[str, int]] = None,
) -> tuple[DimLike, ...]:
    bindings = bindings or {}
    src = _product_eval(source, bindings)
    dst = _product_eval(target, bindings)
    if src is not None and dst is not None and src != dst:
        raise ShapeSystemError(f"reshape changes element count: {src} != {dst}")
    return tuple(target)


def check_shard(
    shape: Sequence[DimLike],
    mesh_axes: Mapping[str, int],
    shard_map: Mapping[Union[str, int], str],
    bindings: Optional[Mapping[str, int]] = None,
) -> list[ShapeDiagnostic]:
    """Validate that sharded dimensions are divisible by their mesh axis size."""

    bindings = bindings or {}
    dims = tuple(shape)
    errors: list[ShapeDiagnostic] = []
    for dim_ref, axis in shard_map.items():
        if axis not in mesh_axes:
            errors.append(ShapeDiagnostic("shape-shard-axis", f"unknown mesh axis {axis!r}"))
            continue
        dim_obj = _resolve_dim_ref(dims, dim_ref)
        if dim_obj is None:
            errors.append(ShapeDiagnostic("shape-shard-dim", f"unknown sharded dimension {dim_ref!r}"))
            continue
        value = _eval_dim(dim_obj, bindings)
        parts = mesh_axes[axis]
        if value is not None and value % parts != 0:
            errors.append(
                ShapeDiagnostic(
                    "shape-shard-divisible",
                    f"{_dim_label(dim_obj)}={value} is not divisible by mesh axis {axis}={parts}",
                    f"pad {_dim_label(dim_obj)} to {next_multiple(value, parts)} or change mesh layout",
                )
            )
    return errors


def check_schedule_tile(dim: DimLike, tile: int, bindings: Mapping[str, int]) -> ScheduleFeasibility:
    if tile <= 0:
        raise ValueError("tile size must be positive")
    value = _eval_dim(dim, bindings)
    if value is None or value % tile == 0:
        return ScheduleFeasibility(True, dim, tile, actual=value)
    return ScheduleFeasibility(False, dim, tile, actual=value, padded=next_multiple(value, tile))


def next_multiple(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def check_shapes(fn: Callable) -> Callable:
    """Mark a Python function as shape-checked by the Tessera frontend."""

    setattr(fn, "__tessera_check_shapes__", True)
    return fn


def dims_compatible(lhs: DimLike, rhs: DimLike) -> bool:
    if _is_one(lhs) or _is_one(rhs):
        return True
    if isinstance(lhs, int) and isinstance(rhs, int):
        return lhs == rhs
    if isinstance(lhs, Dim) and isinstance(rhs, Dim):
        return lhs.name == rhs.name or (
            lhs.value is not None and rhs.value is not None and lhs.value == rhs.value
        )
    return str(lhs) == str(rhs)


def _broadcast_dim(lhs: DimLike, rhs: DimLike) -> DimLike:
    if _is_one(lhs):
        return rhs
    if _is_one(rhs):
        return lhs
    if dims_compatible(lhs, rhs):
        return lhs
    raise ShapeSystemError(f"cannot broadcast {_dim_label(lhs)} with {_dim_label(rhs)}")


def _is_one(dim_value: DimLike) -> bool:
    return dim_value == 1 or (isinstance(dim_value, Dim) and dim_value.value == 1)


def _eval_dim(dim_value: DimLike, bindings: Mapping[str, int]) -> Optional[int]:
    if isinstance(dim_value, int):
        return dim_value
    if isinstance(dim_value, (Dim, DimProduct)):
        return dim_value.eval(bindings)
    return None


def _product_eval(shape: Iterable[DimLike], bindings: Mapping[str, int]) -> Optional[int]:
    values = [_eval_dim(d, bindings) for d in shape]
    if any(v is None for v in values):
        return None
    return reduce(mul, (v for v in values if v is not None), 1)


def _dim_label(dim_value: DimLike) -> str:
    return str(dim_value)


def _resolve_dim_ref(dims: Sequence[DimLike], ref: Union[str, int]) -> Optional[DimLike]:
    if isinstance(ref, int):
        if -len(dims) <= ref < len(dims):
            return dims[ref]
        return None
    for dim_value in dims:
        if isinstance(dim_value, Dim) and dim_value.name == ref:
            return dim_value
    return None


__all__ = [
    "Dim",
    "DimProduct",
    "Layout",
    "RuntimeShapeWitness",
    "ScheduleFeasibility",
    "Shape",
    "ShapeConstraintGraph",
    "ShapeDiagnostic",
    "ShapeShard",
    "ShapeSystemError",
    "broadcast_shape",
    "check_schedule_tile",
    "check_shapes",
    "check_shard",
    "dim",
    "dims_compatible",
    "matmul_shape",
    "next_multiple",
    "reshape_shape",
    "sym",
]
