"""Core Tessera abstractions."""


class Tensor:
    """Phase 1 stub — will be replaced by DistributedArray in Phase 3."""
    pass


class Module:
    """Phase 1 stub — will be replaced by compiled module in Phase 3."""
    pass


class NumericalPolicy:
    """Phase 1 stub — numerics policy (precision/rounding). Phase 2 feature."""
    pass


__all__ = ["Tensor", "Module", "NumericalPolicy"]
