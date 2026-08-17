"""Typed Schedule→Tile boundaries for the PR #568 reference-tier families.

The public operations remain frontend semantics.  This module owns the first
physical compiler contracts: batched tridiagonal solve and the shared
coalition-lattice butterfly used by all four zeta/Mobius transforms.
"""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass

from .graph_ir import GraphIRModule
from .scheduled_matmul import digest_text, find_tessera_opt, run_tessera_opt


_HASH_RE = re.compile(r'tessera\.schedule_hash = "([0-9a-f]{64})"')


@dataclass(frozen=True)
class ScheduledReferenceTierArtifact:
    graph_ir: str
    schedule_ir: str
    tile_ir: str
    target: str
    architecture: str
    family: str
    algorithm: str
    schedule_digest: str

    @property
    def graph_digest(self) -> str:
        return digest_text(self.graph_ir)

    @property
    def schedule_ir_digest(self) -> str:
        return digest_text(self.schedule_ir)

    @property
    def tile_digest(self) -> str:
        return digest_text(self.tile_ir)

    def validate(self) -> None:
        schedule_name = f"schedule.{self.family}"
        tile_name = f"tile.{self.family}_kernel"
        if self.schedule_ir.count(schedule_name) != 1:
            raise ValueError(f"{self.family} requires one Schedule operation")
        if self.schedule_ir.count(self.schedule_digest) != 3:
            raise ValueError(f"{self.family} has incomplete Schedule identity")
        if self.tile_ir.count(tile_name) != 1:
            raise ValueError(f"{self.family} requires one launch-level Tile operation")
        if "schedule." in self.tile_ir or _HASH_RE.findall(self.tile_ir) != [self.schedule_digest]:
            raise ValueError(f"{self.family} did not cross the Schedule→Tile boundary")
        if f'algorithm = "{self.algorithm}"' not in self.tile_ir and self.family == ("tridiagonal_solve"):
            raise ValueError("tridiagonal algorithm identity was lost")


def lower_scheduled_reference_tier(module: GraphIRModule, *, target: str) -> ScheduledReferenceTierArtifact:
    if target == "x86":
        compiler_target, architecture = "x86", "zen5-avx512"
    elif target == "rocm_gfx1151":
        compiler_target, architecture = "rocm", "gfx1151"
    else:
        raise ValueError("reference-tier physical artifacts support x86 and gfx1151")
    if len(module.functions) != 1 or len(module.functions[0].body) != 1:
        raise ValueError("reference-tier physical lowering requires one Graph operation")
    op_name = module.functions[0].body[0].op_name
    if op_name == "tessera.tridiagonal_solve":
        family = "tridiagonal_solve"
        algorithm = "batch_vector_thomas_v1" if target == "x86" else "cooperative_lds_pcr_v1"
    elif op_name in {
        "tessera.game_subset_zeta",
        "tessera.game_subset_mobius",
        "tessera.game_superset_zeta",
        "tessera.game_superset_mobius",
    }:
        family = "coalition_butterfly"
        algorithm = "ascending_bit_yates_v1"
    else:
        raise ValueError("operation has no shared reference-tier physical contract")

    tool = find_tessera_opt()
    if tool is None:
        raise RuntimeError("reference-tier physical lowering requires tessera-opt")
    targeted = copy.deepcopy(module)
    targeted.module_attrs["tessera.target"] = f'"{compiler_target}"'
    targeted.module_attrs["tessera.arch"] = f'"{architecture}"'
    graph_ir = targeted.to_mlir(target=target, canonical=True)
    schedule_ir = run_tessera_opt(tool, graph_ir, "--tessera-graph-to-schedule")
    tile_ir = run_tessera_opt(tool, schedule_ir, "--tessera-schedule-to-tile")
    hashes = _HASH_RE.findall(tile_ir)
    if len(hashes) != 1:
        raise RuntimeError("reference-tier lowering lost its Schedule digest")
    artifact = ScheduledReferenceTierArtifact(
        graph_ir=graph_ir,
        schedule_ir=schedule_ir,
        tile_ir=tile_ir,
        target=compiler_target,
        architecture=architecture,
        family=family,
        algorithm=algorithm,
        schedule_digest=hashes[0],
    )
    artifact.validate()
    return artifact


__all__ = ["ScheduledReferenceTierArtifact", "lower_scheduled_reference_tier"]
