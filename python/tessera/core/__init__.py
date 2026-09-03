"""Core Tessera abstractions."""


def _as_dtype_token(item):
    """The canonical dtype a subscript element names, or ``None`` if it is a dim.

    Two spellings are documented: a dtype shorthand (``tessera.f32``, a
    ``_DtypeAnnotation`` class or instance -- duck-typed on ``_dtype``/``dtype``
    because ``tessera.core`` is imported before those classes exist) and a
    dtype string (``"bf16"``; aliases such as ``"float32"`` normalize at this
    boundary per Decision #15a).

    Recognized-but-not-first-class spellings are REFUSED by name rather than
    silently demoted to a dimension: ``tf32`` (a math mode, not a storage
    dtype) and the planned/gated set (``uint*``, ``complex*``, ``mxfp*``).
    ``allow_planned_gated`` stays False -- an annotation IS a first-class
    storage declaration, and #15a admits a planned/gated dtype only where
    ``metadata.dtype_status = "planned_gated"`` can be carried, which an
    annotation cannot do. Concretely, most of that set has no element-type
    mapping either (``uint8`` would render ``tensor<?xuint8>``, which MLIR
    rejects), so admitting them here would reintroduce exactly the invalid
    element types ``GRAPH_IR_UNRESOLVED_ELEMENT_TYPE`` exists to stop.
    """
    from ..dtype import _TF32_NOT_A_DTYPE, canonicalize_dtype, is_known_dtype

    if isinstance(item, str):
        # `is_known_dtype` covers canonical + alias + planned/gated; `tf32` is
        # deliberately outside it. Both non-first-class families reach
        # `canonicalize_dtype`, which raises its own named error for each.
        if is_known_dtype(item) or item in _TF32_NOT_A_DTYPE:
            return canonicalize_dtype(item)
        return None
    dtype = getattr(item, "dtype", None) or getattr(item, "_dtype", None)
    return canonicalize_dtype(dtype) if isinstance(dtype, str) else None


class Tensor:
    """Phase 1 stub — will be replaced by DistributedArray in Phase 3."""

    def __class_getitem__(cls, shape):
        """``Tensor["B", "D"]`` / ``Tensor["B", "D", "bf16"]`` / ``Tensor["B", "D", tessera.bf16]``.

        The trailing element may name the storage dtype; it is bound to
        ``dtype`` and excluded from ``__dims__``. A dtype anywhere else is
        refused: a storage dtype is a semantic key (Decision #21a) and is never
        silently read as a dimension name -- that is how ``Tensor["M","K","bf16"]``
        once became a rank-3 tensor with no element type.
        """
        from ..dtype import TesseraDtypeError

        dims = shape if isinstance(shape, tuple) else (shape,)
        dtype = None
        for i, item in enumerate(dims[:-1]):
            if _as_dtype_token(item) is not None:
                raise TesseraDtypeError(
                    f"Tensor[...]: dtype {item!r} at position {i} must be the "
                    f"trailing element -- a storage dtype is a semantic key, "
                    f"not a dimension name (got {dims!r})"
                )
        if dims:
            dtype = _as_dtype_token(dims[-1])
            if dtype is not None:
                dims = dims[:-1]
        label = ", ".join(str(d) for d in dims) + (f", {dtype}" if dtype else "")
        attrs = {"__dims__": dims}
        if dtype is not None:
            attrs["dtype"] = dtype
        return type(f"Tensor[{label}]", (cls,), attrs)


class Module:
    """Phase 1 stub — will be replaced by compiled module in Phase 3."""
    pass


class NumericalPolicy:
    """Phase 1 stub — numerics policy (precision/rounding). Phase 2 feature."""
    pass


__all__ = ["Tensor", "Module", "NumericalPolicy"]
