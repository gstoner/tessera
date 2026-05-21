"""CorrDiff-core benchmark — Sub-5 (2026-05-20).

A minimal weather-diffusion benchmark exercising the stack the Phase 7
asks built:

* ``conv2d`` — NHWC convolution as the regression backbone
* ``attn_local_window_2d`` — 2D local-window attention for spatial bias
* deterministic diffusion noise via ``tessera.rng.RNGKey`` (Philox)
* activation checkpointing via ``tessera.autodiff.checkpoint``
* tiled-field input pipeline (small synthetic 64×64 ERA5-shaped data)

The module is a *reference* implementation (CPU + numpy under the hood)
that produces deterministic outputs.  Native lowering paths come in via
``HaloMeshIntegrationPass`` (sharding) + per-target kernel manifests as
those land.
"""
from .corrdiff_core import (
    CorrDiffConfig,
    CorrDiffModel,
    tile_field,
    diffusion_noise_step,
)
from .benchmark_corrdiff import CorrDiffBenchmark, CorrDiffResult

__all__ = [
    "CorrDiffConfig",
    "CorrDiffModel",
    "tile_field",
    "diffusion_noise_step",
    "CorrDiffBenchmark",
    "CorrDiffResult",
]
