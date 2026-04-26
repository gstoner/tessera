"""
Region privilege type annotations.

Region[mode] is a TYPE ANNOTATION, not a runtime wrapper. It participates in
Python's annotation system via __class_getitem__ and lowers to a
`tessera.effect` attribute on Graph IR function arguments.

Valid modes:
    "read"        — read-only; safe to alias and overlap with other reads
    "write"       — exclusive write; conflicts with any other access on same data
    "reduce_sum"  — parallel reduction by sum; safe to overlap across ranks
    "reduce_max"  — parallel reduction by max
    "reduce_min"  — parallel reduction by min

Design note (do not change):
  Region is a type annotation, not a runtime wrapper. Two @jit functions with
  conflicting Region[write] on overlapping tensors raise TesseraPrivilegeError
  at decoration time, before any kernel is compiled or launched.

IR lowering:
  @jit
  def step(W: Region["read"], Y: Region["write"]):
      ...

  lowers to:
    func.func @step(%W: tensor<...> {tessera.effect = "read"},
                    %Y: tensor<...> {tessera.effect = "write"}) { ... }

Usage:
    from tessera.distributed import Region

    @tessera.jit
    def norm(X: Region["read"], Y: Region["write"]):
        Y[:] = ops.layer_norm(X)

    @tessera.jit
    def grad_step(X: Region["read"], G: Region["reduce_sum"]):
        G += ops.gemm(X, X.T)
"""

from __future__ import annotations
from typing import ClassVar


# Valid privilege modes and their properties
_VALID_MODES = {
    "read":       {"exclusive": False, "reduces": False, "op": None},
    "write":      {"exclusive": True,  "reduces": False, "op": None},
    "reduce_sum": {"exclusive": False, "reduces": True,  "op": "sum"},
    "reduce_max": {"exclusive": False, "reduces": True,  "op": "max"},
    "reduce_min": {"exclusive": False, "reduces": True,  "op": "min"},
}


class RegionType:
    """
    Concrete region privilege annotation object.

    Created via Region["mode"] — do not instantiate directly.

    Attributes:
        mode       : str  — the privilege mode ("read", "write", etc.)
        exclusive  : bool — True if this mode forbids aliasing
        reduces    : bool — True if this mode participates in a reduction
        op         : str | None — reduction op name ("sum", "max", "min") or None
    """

    __slots__ = ("mode", "exclusive", "reduces", "op")

    def __init__(self, mode: str) -> None:
        if mode not in _VALID_MODES:
            raise ValueError(
                f"Unknown region mode {mode!r}. "
                f"Valid modes: {list(_VALID_MODES)}"
            )
        props = _VALID_MODES[mode]
        self.mode: str = mode
        self.exclusive: bool = props["exclusive"]
        self.reduces: bool = props["reduces"]
        self.op: str | None = props["op"]

    def __repr__(self) -> str:
        return f"Region[{self.mode!r}]"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, RegionType) and self.mode == other.mode

    def __hash__(self) -> int:
        return hash(("RegionType", self.mode))

    def to_ir_attr(self) -> str:
        """Return the tessera.effect attribute string for Graph IR emission."""
        return self.mode


class RegionMeta(type):
    """
    Metaclass enabling Region["read"] subscript syntax.

    This makes Region behave like a generic alias without requiring
    actual generic machinery — it is always parameterized by a string.
    """

    _cache: ClassVar[dict[str, RegionType]] = {}

    def __getitem__(cls, mode: str) -> RegionType:
        if mode not in cls._cache:
            cls._cache[mode] = RegionType(mode)
        return cls._cache[mode]

    def __repr__(cls) -> str:
        return "Region"


class Region(metaclass=RegionMeta):
    """
    Privilege annotation for @jit function parameters.

    Usage:
        Region["read"]        — read-only input
        Region["write"]       — exclusive-write output
        Region["reduce_sum"]  — parallel summation (safe across ranks)
        Region["reduce_max"]  — parallel max reduction
        Region["reduce_min"]  — parallel min reduction

    Example:
        @tessera.jit
        def fwd(W: Region["read"], X: Region["read"], Y: Region["write"]):
            Y[:] = ops.gemm(X, W)

    Region is a type annotation, not a runtime wrapper.
    """
    # Instances are never created — all access goes through __getitem__
    pass
