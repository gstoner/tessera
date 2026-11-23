"""Native sm_120 embedding row gather (AR-DECODE-PROGRAM-1 S1c).

The contract lives in C++ (``NativeEmbeddingGather.h``): an isolated
``tessera.gather %table, %tokens {axis = 0}`` over an f32[V, H] table and
i32[B] tokens in an ``nvidia_sm120`` module becomes a content-addressed
``schedule.embedding_gather`` and then ``tile.embedding_gather_kernel``.
Python builds the Graph text, replays the native passes and re-reads every
launch field from the replayed ``tessera.native_contract``; it decides nothing
about the kernel.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import re

from .scheduled_matmul import find_tessera_opt, run_tessera_opt

_CONTRACT_KEYS = frozenset({
    "arch", "bindings", "index_bounds", "index_storage", "schedule", "shape", "storage", "target"})
_SEMANTIC = {
    "target": "nvidia_sm120", "arch": "sm_120", "storage": "f32", "index_storage": "i32",
    "index_bounds": "kernel_checked_status_word", "schedule": "cta_128_per_token_row"}
_KERNEL_KEYS = ("index_bounds", "index_storage", "schedule", "storage")
_CONTRACT_FIELD = re.compile(r'(\w+) = ("[^"]*"|\[[^\]]*\]|array<i64(?:: [-0-9, ]*)?>)(?:, |$)')
_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z_0-9]*")
_INT32_MAX = 2**31 - 1


def _parse_native_contract(text: str) -> dict[str, object]:
    fields: dict[str, object] = {}
    position = 0
    for match in _CONTRACT_FIELD.finditer(text):
        if match.start() != position:
            raise ValueError("embedding gather native contract has an unparsable field")
        key, raw = match.group(1), match.group(2)
        if key in fields:
            raise ValueError("embedding gather native contract repeats a field")
        if raw.startswith('"'):
            fields[key] = raw[1:-1]
        elif raw.startswith("["):
            fields[key] = tuple(json.loads(raw))
        else:
            body = raw.removeprefix("array<i64").removeprefix(": ").removesuffix(">")
            fields[key] = tuple(int(item) for item in body.split(", ")) if body else ()
        position = match.end()
    if position != len(text):
        raise ValueError("embedding gather native contract has an unparsable field")
    return fields


@dataclass(frozen=True)
class ScheduledEmbeddingGatherArtifact:
    """Graph -> Schedule -> Tile replay of one sm_120 embedding row gather.

    :meth:`validate` replays both passes and re-reads every field from the
    replayed native contract.
    """

    graph_ir: str
    schedule_ir: str
    tile_ir: str
    schedule_digest: str
    names: tuple[str, str, str]
    vocab: int
    hidden: int
    batch: int
    entry: str

    def validate(self) -> None:
        tool = find_tessera_opt()
        if tool is None:
            raise RuntimeError("sm_120 embedding gather requires native Schedule replay")
        if run_tessera_opt(tool, self.graph_ir, "--tessera-graph-to-schedule") != self.schedule_ir:
            raise ValueError("embedding gather Schedule IR disagrees with Graph replay")
        if run_tessera_opt(tool, self.schedule_ir, "--tessera-schedule-to-tile") != self.tile_ir:
            raise ValueError("embedding gather Tile IR disagrees with native Schedule replay")
        contracts = re.findall(r"tessera\.native_contract = \{([^{}]*)\}", self.tile_ir)
        functions = re.findall(r"llvm\.func @(\w+)\(([^)]*)\)", self.tile_ir)
        kernels = re.findall(r"tile\.embedding_gather_kernel ([^{]*)\{([^}]*)\}", self.tile_ir)
        if len(contracts) != 1 or len(functions) != 1 or len(kernels) != 1:
            raise ValueError("embedding gather Tile IR must carry one native contract and kernel")
        fields = _parse_native_contract(contracts[0])
        if set(fields) != _CONTRACT_KEYS:
            raise ValueError("embedding gather native contract has unexpected fields")
        if any(fields[key] != value for key, value in _SEMANTIC.items()):
            raise ValueError("embedding gather native contract leaves the sm_120 envelope")
        if (fields["bindings"] != tuple(self.names)
                or fields["shape"] != (self.vocab, self.hidden, self.batch)):
            raise ValueError("embedding gather artifact fields disagree with its native Tile contract")
        entry, parameters = functions[0]
        operands, attr_text = kernels[0]
        attrs = dict(re.findall(r'(\w+) = "([^"]*)"', attr_text))
        if (entry != self.entry or parameters.count(": !llvm.ptr") != 4
                or parameters.count(": i64") != 3 or operands.count("%") != 7
                or attrs != {key: str(fields[key]) for key in _KERNEL_KEYS}):
            raise ValueError("embedding gather Tile kernel disagrees with its runtime ABI")
        if re.findall(r'tessera.schedule_hash = "([0-9a-f]{64})"', self.tile_ir) != [self.schedule_digest]:
            raise ValueError("embedding gather schedule hash disagrees")
        if self.entry != f"tessera_tile_embedding_gather_f32_{self.schedule_digest[:10]}":
            raise ValueError("embedding gather entry symbol disagrees with its schedule digest")


def lower_scheduled_embedding_gather(
    names: tuple[str, str, str], *, vocab: int, hidden: int, batch: int,
) -> ScheduledEmbeddingGatherArtifact:
    """Replay one isolated sm_120 ``tessera.gather`` of table rows through native passes."""
    if (len(names) != 3 or len(set(names)) != 3 or "status" in names
            or any(not _IDENTIFIER.fullmatch(n) for n in names)):
        raise ValueError("sm_120 embedding gather bindings must be three unique identifiers, none 'status'")
    if any(type(d) is not int or d <= 0 for d in (vocab, hidden, batch)):
        raise ValueError("sm_120 embedding gather extents must be positive integers")
    if vocab > _INT32_MAX or batch > _INT32_MAX:
        raise ValueError("sm_120 embedding gather extents exceed its launch ABI")
    table_t = f"tensor<{vocab}x{hidden}xf32>"
    tokens_t = f"tensor<{batch}xi32>"
    result_t = f"tensor<{batch}x{hidden}xf32>"
    graph = f"""module attributes {{tessera.target = "nvidia_sm120", tessera.arch = "sm_120"}} {{
      func.func @embedding_gather(%table: {table_t}, %tokens: {tokens_t}) -> {result_t}
          attributes {{tessera.bindings = {json.dumps(list(names))}}} {{
        %rows = tessera.gather %table, %tokens {{axis = 0 : i64}} : ({table_t}, {tokens_t}) -> {result_t}
        return %rows : {result_t}
      }}
    }}"""
    tool = find_tessera_opt()
    if tool is None:
        raise RuntimeError("sm_120 embedding gather requires production tessera-opt")
    schedule = run_tessera_opt(tool, graph, "--tessera-graph-to-schedule")
    tile = run_tessera_opt(tool, schedule, "--tessera-schedule-to-tile")
    hashes = re.findall(r'tessera.schedule_hash = "([0-9a-f]{64})"', tile)
    if len(hashes) != 1:
        raise RuntimeError("sm_120 embedding gather lowering lost its unique schedule hash")
    artifact = ScheduledEmbeddingGatherArtifact(
        graph_ir=graph, schedule_ir=schedule, tile_ir=tile, schedule_digest=hashes[0],
        names=(names[0], names[1], names[2]), vocab=vocab, hidden=hidden, batch=batch,
        entry=f"tessera_tile_embedding_gather_f32_{hashes[0][:10]}",
    )
    artifact.validate()
    return artifact
