"""Core Tessera abstractions."""


def _as_dtype_token(item):
    """The canonical dtype a subscript element names, or ``None`` if it is a dim.

    Two spellings are documented: a dtype shorthand (``tessera.f32``, a
    ``_DtypeAnnotation`` class or instance -- duck-typed on ``_dtype``/``dtype``
    because ``tessera.core`` is imported before those classes exist) and a
    dtype string (``"bf16"``; aliases such as ``"float32"`` normalize at this
    boundary per Decision #15a). ``tf32`` is refused by ``canonicalize_dtype``
    -- it is a math mode, not a storage dtype.
    """
    from ..dtype import _TF32_NOT_A_DTYPE, canonicalize_dtype, is_known_dtype

    if isinstance(item, str):
        # ``tf32`` is dtype-shaped but deliberately not "known": it must be
        # REFUSED by name (canonicalize_dtype raises its dedicated error), never
        # demoted to a dimension called "tf32".
        if is_known_dtype(item) or item in _TF32_NOT_A_DTYPE:
            return canonicalize_dtype(item, allow_planned_gated=True)
        return None
    dtype = getattr(item, "dtype", None) or getattr(item, "_dtype", None)
    return canonicalize_dtype(dtype, allow_planned_gated=True) if isinstance(dtype, str) else None


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
