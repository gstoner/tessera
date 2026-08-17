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
from typing import Mapping, Sequence


class LayoutAlgebraUnavailableError(RuntimeError):
    """The required native layout algebra cannot be loaded."""


class LayoutAlgebraError(ValueError):
    """A layout expression is malformed, unresolved, or out of range."""


@dataclass(frozen=True)
class RearrangePlan:
    expanded_shape: tuple[int, ...]
    permutation: tuple[int, ...]
    output_shape: tuple[int, ...]


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
        build_dirs.extend(root / name for name in ("build", "build-rocm", "build-x86"))
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
    for symbol in (
        "tessera_layout_size_v1",
        "tessera_layout_cosize_v1",
        "tessera_layout_crd2idx_v1",
        "tessera_layout_idx2crd_v1",
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


def idx2crd(shape: Sequence[int], index: int) -> tuple[int, ...]:
    values = tuple(shape)
    output = (ctypes.c_int64 * len(values))()
    status = _library().tessera_layout_idx2crd_v1(
        _i64_array(values), len(values), int(index), output, len(values)
    )
    if status:
        raise LayoutAlgebraError(f"layout inverse coordinate failed with native status {status}")
    return tuple(output)


__all__ = [
    "LayoutAlgebraError",
    "LayoutAlgebraUnavailableError",
    "RearrangePlan",
    "cosize",
    "crd2idx",
    "idx2crd",
    "native_available",
    "rearrange_plan",
    "rearrange_shape_plan",
    "size",
]
