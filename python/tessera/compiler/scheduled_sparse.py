"""The checked 2:4 sparse half matmul as a scheduled native artifact.

Public admission for the gfx1201 SWMMAC stack (2026-09-18): the same
Graph->Schedule->Tile boundaries the dense scheduled families cross, with the
Schedule level owning the kernel the compiler string-builds from the logical
matmul (`NativeSparse.h`: 2:4 index selection and validity words per 16x16
tile, ``schedule.sparse_mma`` -> ``tile.sparse_mma`` -> ``tessera_rocm.swmmac``).
Replay is mandatory and fails closed; the packager (`rocm_native.package_sparse_matmul`)
and the runtime's launch ABI consume this artifact, and a tile whose A block
is not 2:4 sparse refuses the whole launch rather than returning numbers.
AD stays on the logical function (`native_backward`), never on the packing.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .graph_ir import GraphIRModule
from .scheduled_matmul import digest_text, find_tessera_opt, run_tessera_opt

#: The kernel symbol the Schedule-level builder emits.
SPARSE_ENTRY = "probe"
SELECTIONS = ("checked_2to4", "auto_2to4")


def _projection(schedule_ir: str) -> tuple[str, tuple[int, int, int], str, str]:
    selection = re.search(r'tessera\.sparse_selection = "(checked_2to4|auto_2to4)"', schedule_ir)
    shape = re.search(r"tessera\.sparse_shape = array<i64: (\d+), (\d+), (\d+)>", schedule_ir)
    storage = re.search(r'tessera\.sparse_storage = "(f16|bf16)"', schedule_ir)
    output = re.search(r'tessera\.sparse_output = "(f16|bf16|f32)"', schedule_ir)
    if selection is None or shape is None or storage is None or output is None:
        raise ValueError("native sparse descriptor is missing from the Schedule artifact")
    m, n, k = (int(v) for v in shape.groups())
    return selection[1], (m, n, k), storage[1], output[1]


@dataclass(frozen=True)
class ScheduledSparseMatmulArtifact:
    graph_ir: str
    schedule_ir: str
    tile_ir: str
    selection: str
    m: int
    n: int
    k: int
    storage: str
    output: str
    a_name: str
    b_name: str
    output_name: str
    schedule_digest: str
    tile_digest: str
    target: str = "rocm"
    architecture: str = "gfx1201"
    function_name: str = SPARSE_ENTRY

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.m, self.n, self.k)

    @property
    def tiles(self) -> int:
        return (self.m // 16) * (self.n // 16)

    def validate(self) -> None:
        tool = find_tessera_opt()
        if tool is None:
            raise RuntimeError("scheduled sparse matmul validation requires production tessera-opt")
        if self.selection not in SELECTIONS or self.storage not in ("f16", "bf16") or self.output not in ("f16", "bf16", "f32"):
            raise ValueError("scheduled sparse matmul carries an unknown selection or storage")
        if _projection(self.schedule_ir) != (self.selection, self.shape, self.storage, self.output):
            raise ValueError("scheduled sparse matmul descriptor disagrees with its Schedule artifact")
        if self.m % 16 or self.n % 16 or self.k % 32 or max(self.m, self.n, self.k) > 256:
            raise ValueError("scheduled sparse matmul extents are outside the 16/16/32-tiled <=256 envelope")
        if digest_text(self.schedule_ir) != self.schedule_digest or digest_text(self.tile_ir) != self.tile_digest:
            raise ValueError("scheduled sparse matmul artifact identity disagrees")
        if run_tessera_opt(tool, self.schedule_ir, "--tessera-schedule-to-tile") != self.tile_ir:
            raise ValueError("scheduled sparse matmul replay disagrees with the recorded Tile artifact")
        if "tile.sparse_mma" not in self.tile_ir or f"gpu.func @{self.function_name}(" not in self.tile_ir:
            raise ValueError("scheduled sparse matmul Tile artifact lost its sparse MMA or entry")
        if f"memref<{self.tiles * 32}xi32>" not in self.tile_ir:
            raise ValueError("scheduled sparse matmul Tile artifact lost its validity words")


def lower_scheduled_sparse_matmul(module: GraphIRModule, *, selection: str = "checked_2to4") -> ScheduledSparseMatmulArtifact:
    """Admit one logical half matmul and lower it through the production
    boundaries into the checked 2:4 Schedule and Tile artifacts."""
    from .sparse_capture import native_sparse_source

    if selection not in SELECTIONS:
        raise ValueError("unknown sparse selection policy")
    graph_ir = native_sparse_source(module, selection)
    tool = find_tessera_opt()
    if tool is None:
        raise RuntimeError("scheduled sparse matmul lowering requires production tessera-opt")
    schedule_ir = run_tessera_opt(tool, graph_ir, "--tessera-graph-to-schedule")
    tile_ir = run_tessera_opt(tool, schedule_ir, "--tessera-schedule-to-tile")
    projected, (m, n, k), storage, output = _projection(schedule_ir)
    if projected != selection:
        raise ValueError("native sparse selection disagrees with the requested policy")
    fn = module.functions[0]
    output_name = fn.body[0].result or fn.return_values[0].removeprefix("%")
    artifact = ScheduledSparseMatmulArtifact(
        graph_ir=graph_ir, schedule_ir=schedule_ir, tile_ir=tile_ir, selection=selection,
        m=m, n=n, k=k, storage=storage, output=output,
        a_name=fn.args[0].name, b_name=fn.args[1].name, output_name=output_name,
        schedule_digest=digest_text(schedule_ir), tile_digest=digest_text(tile_ir))
    artifact.validate()
    return artifact


__all__ = ["SPARSE_ENTRY", "SELECTIONS", "ScheduledSparseMatmulArtifact", "lower_scheduled_sparse_matmul"]
