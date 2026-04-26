"""
tessera.distributed — Python-level distributed programming abstractions.

This package provides the user-facing API for distributed tensor programming:
  - Region[mode]           — privilege annotations (read/write/reduce)
  - tessera.domain.Rect    — logical iteration space
  - tessera.dist.Block     — placement strategy (how domain maps to mesh)
  - tessera.array          — distributed array backed by a ShardSpec
  - tessera.index_launch   — fan out a kernel across mesh partitions

These objects lower into Schedule IR (schedule.mesh.define, mesh.region,
optimizer.shard) during @jit compilation. They carry no GPU runtime
dependency — Phase 1 targets CPU via the x86 AMX backend.

Build order:
  1. region.py   → Region type annotation object
  2. shard.py    → MeshSpec, ShardSpec
  3. domain.py   → Rect, Block, Cyclic, Replicated
  4. array.py    → DistributedArray.from_domain(), .parts()
  5. launch.py   → @kernel, index_launch()
"""

from .region import Region
from .shard import ShardSpec, MeshSpec
from .domain import Rect, Block, Cyclic, Replicated
from .array import DistributedArray
from .launch import index_launch, kernel

# Namespace aliases matching the programming guide API:
#   tessera.domain.Rect(...)
#   tessera.dist.Block(...)
#   tessera.array.from_domain(...)
import types

domain = types.SimpleNamespace(
    Rect=Rect,
)

dist = types.SimpleNamespace(
    Block=Block,
    Cyclic=Cyclic,
    Replicated=Replicated,
)

array = types.SimpleNamespace(
    from_domain=DistributedArray.from_domain,
)

__all__ = [
    "Region",
    "ShardSpec",
    "MeshSpec",
    "Rect",
    "Block",
    "Cyclic",
    "Replicated",
    "DistributedArray",
    "index_launch",
    "kernel",
    "domain",
    "dist",
    "array",
]
