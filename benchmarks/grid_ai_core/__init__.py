"""Tiny generic gridded-AI core benchmark.

This package intentionally avoids application vocabulary.  It exercises the
generic compiler/runtime substrate needed by regional high-resolution AI models:
tiled fields, local stencil features, 2D local-window attention, fused conv
blocks, deterministic noise, and halo oracle checks.
"""

from .core import (
    GridAICoreBenchmark,
    GridAICoreConfig,
    GridAICoreModel,
    GridAICoreResult,
    deterministic_noise_step,
    local_stencil_feature,
    periodic_halo_oracle,
    tile_field,
)

__all__ = [
    "GridAICoreBenchmark",
    "GridAICoreConfig",
    "GridAICoreModel",
    "GridAICoreResult",
    "deterministic_noise_step",
    "local_stencil_feature",
    "periodic_halo_oracle",
    "tile_field",
]
