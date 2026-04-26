"""Core Tessera abstractions."""


class Tensor:
    """Phase 1 stub — will be replaced by DistributedArray in Phase 3."""

    def __class_getitem__(cls, shape):
        """Allow Tensor["B", "D"] annotation syntax for type hints."""
        dims = shape if isinstance(shape, tuple) else (shape,)
        ann = type(f"Tensor[{', '.join(str(d) for d in dims)}]",
                   (cls,), {"__dims__": dims})
        return ann


class Module:
    """Phase 1 stub — will be replaced by compiled module in Phase 3."""
    pass


class NumericalPolicy:
    """Phase 1 stub — numerics policy (precision/rounding). Phase 2 feature."""
    pass


__all__ = ["Tensor", "Module", "NumericalPolicy"]
