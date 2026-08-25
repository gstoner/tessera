"""ctypes binding to Tessera's single C++ layout-algebra authority.

There is intentionally no Python implementation fallback.  Layout semantics
affect generated addresses, so an absent or stale native symbol is a compiler
configuration error rather than permission to use a second implementation.
"""

from __future__ import annotations

import ctypes
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


class LayoutAlgebraUnavailableError(RuntimeError):
    """The required native layout algebra cannot be loaded."""


class LayoutAlgebraError(ValueError):
    """A layout expression is malformed, unresolved, or out of range."""


LayoutTree = int | tuple["LayoutTree", ...]


@dataclass(frozen=True)
class NestedLayout:
    """A structured nested ``(shape, stride)`` layout value.

    Tuples are hierarchy, not a textual layout DSL.  ``-1`` is the ABI's
    dynamic-leaf sentinel.  Canonicalization transports a dynamic residue
    unchanged; operations whose proof needs its concrete value fail closed.
    """

    shape: LayoutTree
    stride: LayoutTree


@dataclass(frozen=True)
class SlicedLayout:
    """A residual layout plus the address offset from fixed coordinates."""

    layout: NestedLayout
    offset: int


@dataclass(frozen=True)
class RearrangePlan:
    expanded_shape: tuple[int, ...]
    permutation: tuple[int, ...]
    output_shape: tuple[int, ...]


@dataclass(frozen=True)
class LayoutFactorizationProof:
    factorizes: bool
    read_cosize: int
    partition_cosize: int


@dataclass(frozen=True)
class LayoutResidencyProof:
    admitted: bool
    elements: int
    bytes: int
    capacity_bytes: int


class _Node(ctypes.Structure):
    _fields_ = [("value", ctypes.c_int64), ("child_count", ctypes.c_uint32)]


class _Slice(ctypes.Structure):
    _fields_ = [("offset", ctypes.c_int64)]


class _Factorization(ctypes.Structure):
    _fields_ = [
        ("factorizes", ctypes.c_int),
        ("read_cosize", ctypes.c_int64),
        ("partition_cosize", ctypes.c_int64),
    ]


class _Residency(ctypes.Structure):
    _fields_ = [
        ("admitted", ctypes.c_int),
        ("elements", ctypes.c_int64),
        ("bytes", ctypes.c_int64),
        ("capacity_bytes", ctypes.c_int64),
    ]


class _Rank2IndexPlan(ctypes.Structure):
    _fields_ = [
        ("major_coordinate", ctypes.c_uint8),
        ("minor_coordinate", ctypes.c_uint8),
    ]


_LIB: ctypes.CDLL | None = None
_LOAD_ERROR: str | None = None


def _candidate_libraries() -> tuple[Path, ...]:
    root = Path(__file__).resolve().parents[3]
    names = (
        "libtessera_layout_algebra.so",
        "libtessera_layout_algebra.dylib",
        "tessera_layout_algebra.dll",
    )
    candidates: list[Path] = []
    configured = os.environ.get("TESSERA_LAYOUT_ALGEBRA_LIB")
    if configured:
        return (Path(configured).expanduser().resolve(),)
    build_dirs = []
    if selected := os.environ.get("TESSERA_BUILD_DIR"):
        build_dirs.append(Path(selected).expanduser())
    else:
        build_dirs.extend(
            root / name
            for name in ("build", "build-apple", "build-rocm", "build-x86")
        )
    for build in build_dirs:
        for name in names:
            candidates.extend(
                (
                    build / "src/compiler/layout_algebra" / name,
                    build / "lib" / name,
                    build / name,
                )
            )
    # Stable order with duplicates removed.
    return tuple(dict.fromkeys(path.resolve() for path in candidates))


def _configure(lib: ctypes.CDLL) -> ctypes.CDLL:
    i64p = ctypes.POINTER(ctypes.c_int64)
    lib.tessera_layout_algebra_version_v1.argtypes = []
    lib.tessera_layout_algebra_version_v1.restype = ctypes.c_char_p
    lib.tessera_layout_rearrange_plan_v1.argtypes = [
        ctypes.c_char_p,
        i64p,
        ctypes.c_size_t,
        ctypes.c_char_p,
        i64p,
        ctypes.c_size_t,
        i64p,
        ctypes.c_size_t,
        i64p,
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_char_p,
        ctypes.c_size_t,
    ]
    lib.tessera_layout_rearrange_plan_v1.restype = ctypes.c_int
    lib.tessera_layout_size_v1.argtypes = [i64p, ctypes.c_size_t, i64p]
    lib.tessera_layout_cosize_v1.argtypes = [i64p, i64p, ctypes.c_size_t, i64p]
    lib.tessera_layout_crd2idx_v1.argtypes = [
        i64p,
        i64p,
        i64p,
        ctypes.c_size_t,
        i64p,
    ]
    lib.tessera_layout_idx2crd_v1.argtypes = [
        i64p,
        ctypes.c_size_t,
        ctypes.c_int64,
        i64p,
        ctypes.c_size_t,
    ]
    lib.tessera_layout_factorizes_v1.argtypes = [
        i64p, i64p, ctypes.c_size_t,
        i64p, i64p, ctypes.c_size_t,
        ctypes.c_int64, ctypes.POINTER(_Factorization),
    ]
    lib.tessera_layout_residency_v1.argtypes = [
        i64p, i64p, ctypes.c_size_t, ctypes.c_int64, ctypes.c_int64,
        ctypes.POINTER(_Residency),
    ]
    lib.tessera_layout_rank2_index_plan_v1.argtypes = [
        ctypes.c_int,
        ctypes.POINTER(_Rank2IndexPlan),
    ]
    lib.tessera_layout_rank2_index_plan_v1.restype = ctypes.c_int
    nodep = ctypes.POINTER(_Node)
    nested_args: list[Any] = [
        nodep, ctypes.c_size_t, nodep, ctypes.c_size_t
    ]
    nested_out: list[Any] = [
        nodep, ctypes.c_size_t, nodep, ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t),
    ]
    lib.tessera_layout_coalesce_v1.argtypes = nested_args + nested_out
    lib.tessera_layout_right_inverse_v1.argtypes = nested_args + nested_out
    lib.tessera_layout_left_inverse_v1.argtypes = nested_args + nested_out
    lib.tessera_layout_complement_v1.argtypes = (
        nested_args + [ctypes.c_int64] + nested_out
    )
    lib.tessera_layout_product_v1.argtypes = (
        nested_args + nested_args + [ctypes.c_int] + nested_out
    )
    lib.tessera_layout_divide_v1.argtypes = (
        nested_args + nested_args + [ctypes.c_int] + nested_out
    )
    lib.tessera_layout_slice_v1.argtypes = (
        nested_args
        + [i64p, ctypes.c_size_t]
        + nested_out
        + [ctypes.POINTER(_Slice)]
    )
    lib.tessera_layout_compose_v1.argtypes = nested_args + nested_args + nested_out
    for symbol in (
        "tessera_layout_size_v1",
        "tessera_layout_cosize_v1",
        "tessera_layout_crd2idx_v1",
        "tessera_layout_idx2crd_v1",
        "tessera_layout_factorizes_v1",
        "tessera_layout_residency_v1",
        "tessera_layout_coalesce_v1",
        "tessera_layout_right_inverse_v1",
        "tessera_layout_left_inverse_v1",
        "tessera_layout_complement_v1",
        "tessera_layout_product_v1",
        "tessera_layout_divide_v1",
        "tessera_layout_slice_v1",
        "tessera_layout_compose_v1",
    ):
        getattr(lib, symbol).restype = ctypes.c_int
    version = lib.tessera_layout_algebra_version_v1()
    if version != b"tessera.layout_algebra.v1":
        raise LayoutAlgebraUnavailableError(
            f"native layout algebra has incompatible ABI {version!r}"
        )
    return lib


def _library() -> ctypes.CDLL:
    global _LIB, _LOAD_ERROR
    if _LIB is not None:
        return _LIB
    failures: list[str] = []
    for path in _candidate_libraries():
        if not path.is_file():
            continue
        try:
            _LIB = _configure(ctypes.CDLL(str(path), mode=ctypes.RTLD_LOCAL))
            return _LIB
        except (OSError, AttributeError, LayoutAlgebraUnavailableError) as exc:
            failures.append(f"{path}: {exc}")
    _LOAD_ERROR = "; ".join(failures) or "no candidate library exists"
    raise LayoutAlgebraUnavailableError(
        "Tessera layout algebra is unavailable or stale; build target "
        "tessera_layout_algebra in the selected TESSERA_BUILD_DIR or set "
        f"TESSERA_LAYOUT_ALGEBRA_LIB. Details: {_LOAD_ERROR}"
    )


def native_available() -> bool:
    try:
        _library()
    except LayoutAlgebraUnavailableError:
        return False
    return True


def _rearrange_plan(
    input_shape: Sequence[int],
    spec: str,
    *,
    axes_lengths: Mapping[str, int] | None = None,
    allow_dynamic: bool,
) -> RearrangePlan:
    lib = _library()
    shape = tuple(int(extent) for extent in input_shape)
    if not shape or any(extent == 0 or extent < -1 for extent in shape):
        raise LayoutAlgebraError(
            "rearrange input shape must use positive extents or -1 for dynamic"
        )
    if not allow_dynamic and any(extent < 0 for extent in shape):
        raise LayoutAlgebraError("runtime rearrange input extents must be positive")
    # The native parser reports exact ranks, but the caller must provide its
    # buffers up front. One slot per expression byte is a strict upper bound on
    # the number of axis identifiers and avoids a Python-side parser.
    capacity = max(8, len(spec))
    Shape = ctypes.c_int64 * len(shape)
    Buffer = ctypes.c_int64 * capacity
    expanded, permutation, output = Buffer(), Buffer(), Buffer()
    atomic_rank, output_rank = ctypes.c_size_t(), ctypes.c_size_t()
    error = ctypes.create_string_buffer(512)
    bindings = ",".join(
        f"{name}={int(extent)}"
        for name, extent in sorted((axes_lengths or {}).items())
    )
    status = lib.tessera_layout_rearrange_plan_v1(
        spec.encode("utf-8"),
        Shape(*shape),
        len(shape),
        bindings.encode("utf-8"),
        expanded,
        capacity,
        permutation,
        capacity,
        output,
        capacity,
        ctypes.byref(atomic_rank),
        ctypes.byref(output_rank),
        error,
        len(error),
    )
    if status:
        message = error.value.decode("utf-8", errors="replace") or f"native status {status}"
        raise LayoutAlgebraError(f"invalid rearrange layout {spec!r}: {message}")
    return RearrangePlan(
        tuple(expanded[: atomic_rank.value]),
        tuple(permutation[: atomic_rank.value]),
        tuple(output[: output_rank.value]),
    )


def rearrange_plan(
    input_shape: Sequence[int],
    spec: str,
    *,
    axes_lengths: Mapping[str, int] | None = None,
) -> RearrangePlan:
    """Build an executable plan for a fully materialized input shape."""

    return _rearrange_plan(
        input_shape, spec, axes_lengths=axes_lengths, allow_dynamic=False
    )


def rearrange_shape_plan(
    input_shape: Sequence[int],
    spec: str,
    *,
    axes_lengths: Mapping[str, int] | None = None,
) -> RearrangePlan:
    """Infer output rank/extents while preserving unknown dimensions as -1."""

    return _rearrange_plan(
        input_shape, spec, axes_lengths=axes_lengths, allow_dynamic=True
    )


def _i64_array(values: Sequence[int]) -> ctypes.Array[ctypes.c_int64]:
    value_tuple = tuple(int(value) for value in values)
    return (ctypes.c_int64 * len(value_tuple))(*value_tuple)


def _flatten_tree(tree: LayoutTree) -> tuple[int, ...]:
    if isinstance(tree, int):
        return (tree,)
    return tuple(leaf for child in tree for leaf in _flatten_tree(child))


def size(shape: Sequence[int]) -> int:
    values = tuple(shape)
    result = ctypes.c_int64()
    status = _library().tessera_layout_size_v1(
        _i64_array(values), len(values), ctypes.byref(result)
    )
    if status:
        raise LayoutAlgebraError(f"layout size failed with native status {status}")
    return result.value


def cosize(shape: Sequence[int], stride: Sequence[int]) -> int:
    shape_values, stride_values = tuple(shape), tuple(stride)
    if len(shape_values) != len(stride_values):
        raise LayoutAlgebraError("shape and stride ranks must match")
    result = ctypes.c_int64()
    status = _library().tessera_layout_cosize_v1(
        _i64_array(shape_values),
        _i64_array(stride_values),
        len(shape_values),
        ctypes.byref(result),
    )
    if status:
        raise LayoutAlgebraError(f"layout cosize failed with native status {status}")
    return result.value


def crd2idx(shape: Sequence[int], stride: Sequence[int], coord: Sequence[int]) -> int:
    shape_values, stride_values, coords = tuple(shape), tuple(stride), tuple(coord)
    if len(shape_values) != len(stride_values) or len(shape_values) != len(coords):
        raise LayoutAlgebraError("shape, stride, and coordinate ranks must match")
    result = ctypes.c_int64()
    status = _library().tessera_layout_crd2idx_v1(
        _i64_array(shape_values),
        _i64_array(stride_values),
        _i64_array(coords),
        len(shape_values),
        ctypes.byref(result),
    )
    if status:
        raise LayoutAlgebraError(f"layout coordinate failed with native status {status}")
    return result.value


def rank2_index_expression(
    row: str,
    column: str,
    leading_dimension: str,
    *,
    order: str = "row_major",
) -> str:
    """Emit a rank-2 linear-index expression from the native mapping plan.

    This function deliberately performs no Python-side layout decision.  It
    only substitutes caller-owned source expressions into the coordinate order
    returned by :file:`Rank2Index.h` through the versioned native ABI.
    """

    orders = {"row_major": 0, "column_major": 1, "col_major": 1}
    if order not in orders:
        raise LayoutAlgebraError(f"unsupported rank-2 order {order!r}")
    plan = _Rank2IndexPlan()
    status = _library().tessera_layout_rank2_index_plan_v1(
        orders[order], ctypes.byref(plan)
    )
    if status:
        raise LayoutAlgebraError(
            f"rank-2 index planning failed with native status {status}"
        )
    coordinates = (str(row), str(column))
    return (
        f"{coordinates[plan.major_coordinate]} * {leading_dimension} + "
        f"{coordinates[plan.minor_coordinate]}"
    )


def idx2crd(shape: Sequence[int], index: int) -> tuple[int, ...]:
    values = tuple(shape)
    output = (ctypes.c_int64 * len(values))()
    status = _library().tessera_layout_idx2crd_v1(
        _i64_array(values), len(values), int(index), output, len(values)
    )
    if status:
        raise LayoutAlgebraError(f"layout inverse coordinate failed with native status {status}")
    return tuple(output)


def factorizes(
    read: NestedLayout,
    partition: NestedLayout,
    *,
    enumeration_limit: int = 1_000_000,
) -> LayoutFactorizationProof:
    """Prove ``read ⊑ partition`` through the single C++ authority."""

    read_shape = _flatten_tree(read.shape)
    read_stride = _flatten_tree(read.stride)
    partition_shape = _flatten_tree(partition.shape)
    partition_stride = _flatten_tree(partition.stride)
    if len(read_shape) != len(read_stride) or len(partition_shape) != len(
        partition_stride
    ):
        raise LayoutAlgebraError("layout shape/stride profiles must match")
    result = _Factorization()
    status = _library().tessera_layout_factorizes_v1(
        _i64_array(read_shape), _i64_array(read_stride), len(read_shape),
        _i64_array(partition_shape), _i64_array(partition_stride),
        len(partition_shape), int(enumeration_limit), ctypes.byref(result),
    )
    if status:
        raise LayoutAlgebraError(
            f"native layout factorization failed with status {status}"
        )
    return LayoutFactorizationProof(
        bool(result.factorizes), result.read_cosize, result.partition_cosize
    )


def prove_residency(
    layout: NestedLayout,
    *,
    element_bytes: int,
    capacity_bytes: int,
) -> LayoutResidencyProof:
    """Prove a physical materialization footprint from layout ``cosize``."""

    shape = _flatten_tree(layout.shape)
    stride = _flatten_tree(layout.stride)
    if len(shape) != len(stride):
        raise LayoutAlgebraError("layout shape/stride profiles must match")
    result = _Residency()
    status = _library().tessera_layout_residency_v1(
        _i64_array(shape), _i64_array(stride), len(shape), int(element_bytes),
        int(capacity_bytes), ctypes.byref(result),
    )
    if status:
        raise LayoutAlgebraError(
            f"native layout residency proof failed with status {status}"
        )
    return LayoutResidencyProof(
        bool(result.admitted), result.elements, result.bytes,
        result.capacity_bytes,
    )


def _encode_tree(tree: LayoutTree) -> list[_Node]:
    if isinstance(tree, int):
        return [_Node(tree, 0)]
    if not isinstance(tree, tuple) or not tree:
        raise LayoutAlgebraError("layout tree groups must be non-empty tuples")
    nodes = [_Node(0, len(tree))]
    for child in tree:
        nodes.extend(_encode_tree(child))
    return nodes


def _decode_tree(nodes: Sequence[_Node], cursor: int = 0) -> tuple[LayoutTree, int]:
    if cursor >= len(nodes):
        raise LayoutAlgebraError("native layout tree is truncated")
    node = nodes[cursor]
    cursor += 1
    if node.child_count == 0:
        return int(node.value), cursor
    children: list[LayoutTree] = []
    for _ in range(node.child_count):
        child, cursor = _decode_tree(nodes, cursor)
        children.append(child)
    return tuple(children), cursor


def _nested_call(
    symbol: str, *layouts: NestedLayout, scalar_args: tuple[int, ...] = ()
) -> NestedLayout:
    encoded = [_encode_tree(part) for layout in layouts for part in (layout.shape, layout.stride)]
    arrays = [( _Node * len(nodes))(*nodes) for nodes in encoded]
    # Composition can split every input leaf at each A radix boundary.  This
    # conservative capacity stays native-owned data transport, not a Python
    # algebra evaluation.
    capacity = max(8, sum(len(nodes) for nodes in encoded) * 2)
    Output = _Node * capacity
    shape_out, stride_out = Output(), Output()
    shape_count, stride_count = ctypes.c_size_t(), ctypes.c_size_t()
    args: list[object] = []
    for array in arrays:
        args.extend((array, len(array)))
    args.extend(scalar_args)
    args.extend((shape_out, capacity, stride_out, capacity,
                 ctypes.byref(shape_count), ctypes.byref(stride_count)))
    status = getattr(_library(), symbol)(*args)
    if status:
        raise LayoutAlgebraError(f"native {symbol.removeprefix('tessera_layout_').removesuffix('_v1')} failed with status {status}")
    shape, shape_end = _decode_tree(shape_out[: shape_count.value])
    stride, stride_end = _decode_tree(stride_out[: stride_count.value])
    if shape_end != shape_count.value or stride_end != stride_count.value:
        raise LayoutAlgebraError("native layout tree has trailing nodes")
    return NestedLayout(shape, stride)


def coalesce(layout: NestedLayout) -> NestedLayout:
    """Return the native canonical form, preserving any dynamic residue."""

    return _nested_call("tessera_layout_coalesce_v1", layout)


def composition(lhs: NestedLayout, rhs: NestedLayout) -> NestedLayout:
    """Materialize ``lhs(rhs(c))`` through the C++ layout authority."""

    return _nested_call("tessera_layout_compose_v1", lhs, rhs)


def right_inverse(layout: NestedLayout) -> NestedLayout:
    """Return the inverse coordinate layout for the compact bijective subset."""

    return _nested_call("tessera_layout_right_inverse_v1", layout)


def left_inverse(layout: NestedLayout) -> NestedLayout:
    """Return the left inverse for the compact bijective subset."""

    return _nested_call("tessera_layout_left_inverse_v1", layout)


def complement(layout: NestedLayout, cotarget: int | None = None) -> NestedLayout:
    """Manufacture the native gap layout through the optional cotarget."""

    return _nested_call(
        "tessera_layout_complement_v1",
        layout,
        scalar_args=(0 if cotarget is None else int(cotarget),),
    )


_PRODUCT_VARIANTS = {
    "logical": 0,
    "zipped": 1,
    "tiled": 2,
    "flat": 3,
    "blocked": 4,
    "raked": 5,
}
_DIVIDE_VARIANTS = {name: value for name, value in _PRODUCT_VARIANTS.items() if value <= 3}


def product(lhs: NestedLayout, rhs: NestedLayout, *, variant: str = "logical") -> NestedLayout:
    """Build a disjoint-copy product with the requested native grouping."""

    try:
        kind = _PRODUCT_VARIANTS[variant]
    except KeyError as exc:
        raise LayoutAlgebraError(f"unknown product variant {variant!r}") from exc
    return _nested_call("tessera_layout_product_v1", lhs, rhs, scalar_args=(kind,))


def divide(layout: NestedLayout, tiler: NestedLayout, *, variant: str = "logical") -> NestedLayout:
    """Divide by a compact rectangular tiler with the requested grouping."""

    try:
        kind = _DIVIDE_VARIANTS[variant]
    except KeyError as exc:
        raise LayoutAlgebraError(f"unknown divide variant {variant!r}") from exc
    return _nested_call("tessera_layout_divide_v1", layout, tiler, scalar_args=(kind,))


def slice_layout(layout: NestedLayout, coordinates: Sequence[int]) -> SlicedLayout:
    """Fix or retain each flattened mode through the native slice carrier.

    ``-1`` retains a mode. Any other value fixes that mode and becomes part of
    the returned byte-free element offset.
    """

    encoded_shape, encoded_stride = _encode_tree(layout.shape), _encode_tree(layout.stride)
    Shape = _Node * len(encoded_shape)
    Stride = _Node * len(encoded_stride)
    Coordinate = ctypes.c_int64 * len(coordinates)
    shape_out_capacity = max(8, len(encoded_shape) * 2)
    stride_out_capacity = max(8, len(encoded_stride) * 2)
    OutputShape = _Node * shape_out_capacity
    OutputStride = _Node * stride_out_capacity
    shape_out, stride_out = OutputShape(), OutputStride()
    shape_count, stride_count = ctypes.c_size_t(), ctypes.c_size_t()
    result = _Slice()
    status = _library().tessera_layout_slice_v1(
        Shape(*encoded_shape), len(encoded_shape), Stride(*encoded_stride), len(encoded_stride),
        Coordinate(*(int(value) for value in coordinates)), len(coordinates),
        shape_out, shape_out_capacity, stride_out, stride_out_capacity,
        ctypes.byref(shape_count), ctypes.byref(stride_count), ctypes.byref(result),
    )
    if status:
        raise LayoutAlgebraError(f"native slice failed with status {status}")
    shape, shape_end = _decode_tree(shape_out[: shape_count.value])
    stride, stride_end = _decode_tree(stride_out[: stride_count.value])
    if shape_end != shape_count.value or stride_end != stride_count.value:
        raise LayoutAlgebraError("native slice layout tree has trailing nodes")
    return SlicedLayout(NestedLayout(shape, stride), result.offset)


__all__ = [
    "LayoutFactorizationProof",
    "LayoutResidencyProof",
    "LayoutAlgebraError",
    "LayoutAlgebraUnavailableError",
    "LayoutTree",
    "NestedLayout",
    "RearrangePlan",
    "SlicedLayout",
    "cosize",
    "coalesce",
    "complement",
    "composition",
    "crd2idx",
    "divide",
    "factorizes",
    "idx2crd",
    "native_available",
    "product",
    "prove_residency",
    "rank2_index_expression",
    "rearrange_plan",
    "rearrange_shape_plan",
    "left_inverse",
    "right_inverse",
    "size",
    "slice_layout",
]
