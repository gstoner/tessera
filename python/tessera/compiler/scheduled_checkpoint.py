"""Native saved-LSE tensor contracts through Schedule and launch-level Tile IR."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import re
import struct

from .scheduled_matmul import find_tessera_opt, run_tessera_opt


@dataclass(frozen=True)
class ScheduledCheckpointArtifact:
    graph_ir: str
    schedule_ir: str
    tile_ir: str
    entry: str
    schedule_digest: str
    backward: bool
    names: tuple[str, ...]
    dims: tuple[int, ...]
    scale: float
    causal: bool

    def validate(self) -> None:
        if type(self.backward) is not bool or type(self.causal) is not bool:
            raise ValueError("checkpoint role and causal must be boolean")
        if len(self.dims) != 7 or any(type(d) is not int or d <= 0 for d in self.dims):
            raise ValueError("checkpoint dimensions must be positive integers")
        if (
            isinstance(self.scale, bool)
            or not isinstance(self.scale, (int, float))
            or not math.isfinite(self.scale)
            or self.scale <= 0
        ):
            raise ValueError("checkpoint scale must be finite and positive")
        tool = find_tessera_opt()
        if tool is None:
            raise RuntimeError("checkpoint validation requires production tessera-opt")
        if run_tessera_opt(tool, self.schedule_ir, "--tessera-schedule-to-tile") != self.tile_ir:
            raise ValueError("checkpoint Tile IR disagrees with native Schedule replay")
        # The replayed native wrapper owns the descriptor projection. Graph text
        # is retained as provenance only; package consumers never regenerate it.
        count = 5 if self.backward else 3
        fields = {
            "family": json.dumps("attention_checkpoint_backward" if self.backward else "attention_checkpoint_forward"),
            "arguments": json.dumps(self.names[:count]),
            "results": json.dumps(self.names[count:]),
            "shape": "array<i64: " + ", ".join(map(str, self.dims)) + ">",
            "causal": str(self.causal).lower(),
        }
        for key, value in fields.items():
            if f"{key} = {value}" not in self.tile_ir:
                raise ValueError("checkpoint metadata disagrees with native contract")
        scale = re.search(r"scale = ([^ ,}]+) : f32", self.tile_ir)
        if scale is None:
            raise ValueError("checkpoint native scale is missing")
        text = scale[1]
        scale_value = struct.unpack(">d", int(text, 16).to_bytes(8, "big"))[0] if text.startswith("0x") else float(text)
        if struct.pack("f", scale_value) != struct.pack("f", self.scale):
            raise ValueError("checkpoint scale disagrees with native contract")
        if re.findall(r'tessera.schedule_hash = "([0-9a-f]{64})"', self.tile_ir) != [self.schedule_digest]:
            raise ValueError("checkpoint Schedule hash disagrees")
        if re.findall(r"llvm.func @([\w]+)\(", self.tile_ir) != [self.entry]:
            raise ValueError("checkpoint entry disagrees")


def _graph_text(names, dims, scale, causal, backward):
    if len(dims) != 7 or any(type(d) is not int or d <= 0 for d in dims):
        raise ValueError("checkpoint requires positive integer dimensions")
    if type(backward) is not bool or type(causal) is not bool:
        raise ValueError("checkpoint role and causal must be boolean")
    if len(names) != (8 if backward else 5) or any(
        not isinstance(n, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", n) for n in names
    ):
        raise ValueError("checkpoint requires identifier bindings")
    b, hq, hkv, sq, sk, d, dv = dims

    def tensor(shape):
        return "tensor<" + "x".join(map(str, shape)) + "xf32>"

    q, k, v = map(tensor, [(b, hq, sq, d), (b, hkv, sk, d), (b, hkv, sk, dv)])
    o, lse = tensor((b, hq, sq, dv)), tensor((b, hq, sq))
    inputs, outputs = ([o, q, k, v, lse], [q, k, v]) if backward else ([q, k, v], [o, lse])
    count = len(inputs)
    arg_names, result_names = names[:count], names[count:]
    args = ", ".join(f"%arg{i}: {t}" for i, t in enumerate(inputs))
    operands = ", ".join(f"%arg{i}" for i in range(count))
    results = ", ".join(f"%r{i}" for i in range(len(outputs)))
    role = "backward" if backward else "forward"
    return f"""module attributes {{tessera.target = "nvidia_sm120", tessera.arch = "sm_120"}} {{
  func.func @checkpoint({args}) -> ({", ".join(outputs)}) attributes {{
    tessera.argument_bindings = {json.dumps(arg_names)}, tessera.result_bindings = {json.dumps(result_names)}
  }} {{
    {results} = "tessera_attn.checkpoint_{role}"({operands}) {{scale = {scale!r} : f32, causal = {str(causal).lower()}}}
      : ({", ".join(inputs)}) -> ({", ".join(outputs)})
    return {results} : {", ".join(outputs)}
  }}
}}
"""


def lower_scheduled_checkpoint(names, dims, scale, causal, *, backward=False):
    tool = find_tessera_opt()
    if tool is None:
        raise RuntimeError("checkpoint lowering requires production tessera-opt")
    graph = _graph_text(names, dims, scale, causal, backward)
    schedule = run_tessera_opt(tool, graph, "--tessera-graph-to-schedule")
    tile = run_tessera_opt(tool, schedule, "--tessera-schedule-to-tile")
    hashes = re.findall(r'tessera.schedule_hash = "([0-9a-f]{64})"', tile)
    entries = re.findall(r"llvm.func @([\w]+)\(", tile)
    if len(hashes) != 1 or len(entries) != 1:
        raise RuntimeError("native checkpoint lowering lost its unique entry/hash")
    artifact = ScheduledCheckpointArtifact(
        graph, schedule, tile, entries[0], hashes[0], backward, tuple(names), tuple(dims), scale, causal
    )
    artifact.validate()
    return artifact
