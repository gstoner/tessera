"""Native saved-LSE tensor contracts through Schedule and launch-level Tile IR."""

from __future__ import annotations

import copy
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
    bias: bool = False
    bias_gradient: bool = False
    frontend_argument_indices: tuple[int, ...] = ()
    bias_shape: tuple[int, ...] = ()
    gradient_activity: tuple[int, ...] = ()
    compact_gradients: bool = False
    compact_launch: str = "packed_v1"
    compact_threads: int = 128
    lse_cotangent: bool = False
    shape_bounds: tuple[int, ...] = ()

    def validate(self) -> None:
        if type(self.lse_cotangent) is not bool or (self.lse_cotangent and not self.backward):
            raise ValueError("LSE cotangent requires a boolean backward role")
        if type(self.compact_threads) is not int or self.compact_threads not in (64,128) or (not self.compact_gradients and self.compact_threads != 128):
            raise ValueError("compact checkpoint threads require 64 or 128 native threads")
        if self.compact_launch not in ("packed_v1","logical_v1") or (not self.compact_gradients and self.compact_launch != "packed_v1"):
            raise ValueError("compact checkpoint launch requires a compact native layout")
        if type(self.compact_gradients) is not bool or (self.compact_gradients and not self.gradient_activity):
            raise ValueError("compact checkpoint outputs require gradient activity")
        if self.gradient_activity and (
                not self.backward or not isinstance(self.gradient_activity, tuple) or
                len(self.gradient_activity) != 3 + int(self.bias_gradient) or
                any(type(x) is not int or x not in (0, 1) for x in self.gradient_activity) or
                not any(self.gradient_activity)):
            raise ValueError("checkpoint gradient activity requires nonempty binary backward result roles")
        if type(self.bias_gradient) is not bool or (self.bias_gradient and (not self.backward or not self.bias)):
            raise ValueError("bias gradient requires a biased backward checkpoint")
        if type(self.backward) is not bool or type(self.causal) is not bool or type(self.bias) is not bool:
            raise ValueError("checkpoint role and causal must be boolean")
        if len(self.names) != (9 if self.backward else 5) + int(self.bias) + int(self.bias_gradient) + int(self.lse_cotangent):
            raise ValueError("checkpoint metadata binding count disagrees with gradient contract")
        from .attention_shape_contract import attention_dimensions
        attention_dimensions(self.dims,self.shape_bounds)
        if self.bias_shape:
            b,hq,_,sq,sk,_,_ = self.dims
            if (not self.bias or not isinstance(self.bias_shape, tuple) or len(self.bias_shape) != 4 or
                    any(type(d) is not int or (d != 1 and d != n)
                        for d,n in zip(self.bias_shape,(b,hq,sq,sk),strict=True))):
                raise ValueError("checkpoint bias shape must broadcast to logical score dimensions")
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
        if not isinstance(self.graph_ir, str) or not self.graph_ir.strip():
            raise ValueError("checkpoint retained Graph IR is missing")
        if run_tessera_opt(tool, self.graph_ir, "--tessera-graph-to-schedule") != self.schedule_ir:
            raise ValueError("checkpoint Schedule IR disagrees with retained Graph replay")
        if run_tessera_opt(tool, self.schedule_ir, "--tessera-schedule-to-tile") != self.tile_ir:
            raise ValueError("checkpoint Tile IR disagrees with native Schedule replay")
        # Both native replay steps verify ancestry before its hashes are exposed.
        # Consumers use retained source; they never reconstruct a Python Graph.
        physical_contract = re.search(r'tessera.native_contract = \{([^\n]*)\}', self.tile_ir)
        if physical_contract is None:
            raise ValueError("checkpoint native physical contract is missing")
        # Embedded paired lineage may describe the sibling's seed/activity.
        # Validate this executable's policy against its own sealed contract.
        contract_text = physical_contract[1]
        count = (6 if self.backward else 3) + int(self.bias) + int(self.lse_cotangent)
        fields = {
            "family": json.dumps("attention_checkpoint_backward" if self.backward else "attention_checkpoint_forward"),
            "arguments": json.dumps(self.names[:count]),
            "results": json.dumps(self.names[count:]),
            "shape": "array<i64: " + ", ".join(map(str, self.dims)) + ">",
            "causal": str(self.causal).lower(),
            "bias": str(self.bias).lower(),
        }
        if self.shape_bounds:
            fields["shape_bounds"] = "array<i64: " + ", ".join(map(str,self.shape_bounds)) + ">"
            fields["shape_policy"] = json.dumps("bounded_sequences_v1")
        elif "shape_bounds =" in contract_text:
            raise ValueError("checkpoint sequence bounds metadata is missing")
        if self.lse_cotangent:
            fields["lse_cotangent"] = "true"
        elif "lse_cotangent = true" in contract_text:
            raise ValueError("checkpoint LSE cotangent metadata is missing")
        if self.gradient_activity:
            fields["gradient_activity"] = "array<i64: " + ", ".join(map(str, self.gradient_activity)) + ">"
            fields["inactive_gradient"] = json.dumps("absent_v1" if self.compact_gradients else "zero_fill_v1")
            if self.compact_gradients:
                fields["gradient_output"] = json.dumps("compact_v1")
                fields["gradient_launch"] = json.dumps(self.compact_launch)
                fields["gradient_block_threads"] = f"{self.compact_threads} : i64"
                fields["physical_results"] = json.dumps([name for name, active in
                    zip(self.names[count:], self.gradient_activity, strict=True) if active])
            elif "gradient_output =" in contract_text:
                raise ValueError("checkpoint compact gradient output is missing")
        elif "gradient_activity =" in contract_text:
            raise ValueError("checkpoint gradient activity is missing")
        if self.bias_shape:
            fields["bias_shape"] = "array<i64: " + ", ".join(map(str,self.bias_shape)) + ">"
            fields["bias_gradient_reduction"] = json.dumps("physical_owner_lexicographic_bhqk_v1")
        elif "bias_shape =" in contract_text:
            raise ValueError("checkpoint physical bias shape is missing")
        for key, value in fields.items():
            if f"{key} = {value}" not in contract_text:
                raise ValueError("checkpoint metadata disagrees with native contract")
        mapping = self.frontend_argument_indices
        if not isinstance(mapping, tuple):
            raise ValueError("checkpoint frontend mapping must be a tuple")
        if mapping:
            if (not isinstance(mapping, tuple) or any(type(i) is not int for i in mapping) or
                    sorted(mapping) != list(range(3 + int(self.bias)))):
                raise ValueError("checkpoint frontend input mapping must be a permutation")
            text = "frontend_argument_indices = array<i64: " + ", ".join(map(str,mapping)) + ">"
            if text not in contract_text:
                raise ValueError("checkpoint frontend mapping disagrees with native contract")
        elif "frontend_argument_indices =" in contract_text:
            raise ValueError("checkpoint frontend mapping is missing")
        scale = re.search(r"scale = ([^ ,}]+) : f32", contract_text)
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


def _graph_text(names, dims, scale, causal, backward, bias=False, bias_gradient=False, bias_shape=None):
    if len(dims) != 7 or any(type(d) is not int or d <= 0 for d in dims):
        raise ValueError("checkpoint requires positive integer dimensions")
    if type(backward) is not bool or type(causal) is not bool or type(bias) is not bool:
        raise ValueError("checkpoint role and causal must be boolean")
    if type(bias_gradient) is not bool or (bias_gradient and (not bias or not backward)):
        raise ValueError("bias gradient requires a biased backward checkpoint")
    if len(names) != (9 if backward else 5) + int(bias) + int(bias_gradient) or any(
        not isinstance(n, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", n) for n in names
    ):
        raise ValueError("checkpoint requires identifier bindings")
    b, hq, hkv, sq, sk, d, dv = dims

    def tensor(shape):
        return "tensor<" + "x".join(map(str, shape)) + "xf32>"

    physical_bias = tuple(bias_shape) if bias_shape is not None else (b,hq,sq,sk)
    if (bias_shape is not None and (not bias or len(physical_bias) != 4 or
            any(type(x) is not int or (x != 1 and x != n)
                for x,n in zip(physical_bias,(b,hq,sq,sk),strict=True)))):
        raise ValueError("checkpoint bias shape must broadcast to logical score dimensions")
    q, k, v = map(tensor, [(b, hq, sq, d), (b, hkv, sk, d), (b, hkv, sk, dv)])
    o, lse = tensor((b, hq, sq, dv)), tensor((b, hq, sq))
    inputs, outputs = ([o, q, k, v, o, lse], [q, k, v]) if backward else ([q, k, v], [o, lse])
    if bias:
        inputs.insert(5 if backward else 3, tensor(physical_bias))
    if bias_gradient:
        outputs.append(tensor(physical_bias))
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


def lower_scheduled_checkpoint(names, dims, scale, causal, *, backward=False, bias=False, bias_gradient=False, bias_shape=None):
    tool = find_tessera_opt()
    if tool is None:
        raise RuntimeError("checkpoint lowering requires production tessera-opt")
    graph = _graph_text(names, dims, scale, causal, backward, bias, bias_gradient, bias_shape)
    schedule = run_tessera_opt(tool, graph, "--tessera-graph-to-schedule")
    tile = run_tessera_opt(tool, schedule, "--tessera-schedule-to-tile")
    hashes = re.findall(r'tessera.schedule_hash = "([0-9a-f]{64})"', tile)
    entries = re.findall(r"llvm.func @([\w]+)\(", tile)
    if len(hashes) != 1 or len(entries) != 1:
        raise RuntimeError("native checkpoint lowering lost its unique entry/hash")
    artifact = ScheduledCheckpointArtifact(
        graph, schedule, tile, entries[0], hashes[0], backward, tuple(names), tuple(dims), scale, causal, bias, bias_gradient,
        bias_shape=tuple(bias_shape) if bias_shape is not None and tuple(bias_shape) !=
            (dims[0],dims[1],dims[3],dims[4]) else ()
    )
    artifact.validate()
    return artifact


def lower_generated_checkpoint(source: str, *, backward: bool = False, prune_inactive: bool = False, compact_gradients: bool = False, compact_launch: str = "packed_v1", compact_threads: int = 128):
    """Export a fresh native AD checkpoint through the existing Schedule path.

    Source is parser-bound MLIR with the explicit SM120 target. No GraphIRModule
    reconstruction occurs; the native pass owns operand ordering and AD lineage.
    """
    if (type(backward) is not bool or type(prune_inactive) is not bool or type(compact_gradients) is not bool
            or (compact_gradients and not (backward and prune_inactive))):
        raise ValueError('checkpoint role and pruning must be boolean')
    tool = find_tessera_opt()
    if tool is None:
        raise RuntimeError('checkpoint AD export requires production tessera-opt')
    if compact_launch not in ("packed_v1","logical_v1") or (not compact_gradients and compact_launch != "packed_v1"):
        raise ValueError("compact launch requires a compact native layout")
    if type(compact_threads) is not int or compact_threads not in (64,128) or (not compact_gradients and compact_threads != 128):
        raise ValueError("compact checkpoint threads require 64 or 128 native threads")
    role = 'backward' if backward else 'forward'
    options = '--tessera-autodiff-paired=checkpoint-product='+role
    if prune_inactive:
        options += ' prune-checkpoint-gradients=true'
    if compact_gradients:
        options += ' compact-checkpoint-gradients=true compact-checkpoint-launch='+compact_launch+' compact-checkpoint-threads='+str(compact_threads)
    graph = run_tessera_opt(tool, source, options)
    schedule = run_tessera_opt(tool, graph, '--tessera-graph-to-schedule')
    tile = run_tessera_opt(tool, schedule, '--tessera-schedule-to-tile')
    return _decode_checkpoint(graph, schedule, tile, backward=backward,
        require_frontend_mapping=True, compact_gradients=compact_gradients,
        compact_launch=compact_launch, compact_threads=compact_threads)


def lower_checkpoint_graph(module, *, backward=False):
    """Lower the original saved-LSE Graph; native passes own its semantics."""
    if type(backward) is not bool or len(module.functions) != 1:
        raise ValueError("checkpoint requires one function and a boolean role")
    tool = find_tessera_opt()
    if tool is None:
        raise RuntimeError("checkpoint lowering requires production tessera-opt")
    targeted = copy.deepcopy(module)
    fn = targeted.functions[0]
    fn.fn_attrs["tessera.argument_bindings"] = json.dumps([arg.name for arg in fn.args])
    fn.fn_attrs["tessera.result_bindings"] = json.dumps(
        [name.removeprefix("%") for name in fn.return_values])
    targeted.module_attrs["tessera.target"] = '"nvidia_sm120"'
    targeted.module_attrs["tessera.arch"] = '"sm_120"'
    graph = targeted.to_mlir(target="nvidia_sm120", canonical=True)
    schedule = run_tessera_opt(tool, graph, "--tessera-graph-to-schedule")
    tile = run_tessera_opt(tool, schedule, "--tessera-schedule-to-tile")
    return _decode_checkpoint(graph, schedule, tile, backward=backward)


def _decode_checkpoint(graph, schedule, tile, *, backward,
                       require_frontend_mapping=False, compact_gradients=False,
                       compact_launch="packed_v1", compact_threads=128):
    # Read the native physical contract; embedded lineage may itself contain
    # tensor types, so never rediscover dimensions from arbitrary type strings.
    contract = re.search(r'tessera.native_contract = \{([^\n]*)\}', tile)
    if contract is None:
        raise ValueError('native checkpoint export lost its physical contract')
    text = contract[1]
    shape = re.search(r'\bshape = array<i64: ([-0-9, ]+)>',text)
    scale = re.search(r'scale = ([^ ,}]+) : f32',text)
    causal = re.search(r'causal = (true|false)',text)
    if shape is None or scale is None or causal is None:
        raise ValueError('native checkpoint export lost shape or numeric policy')
    value = scale[1]
    scale_value = struct.unpack('>d',int(value,16).to_bytes(8,'big'))[0] if value.startswith('0x') else float(value)
    hashes = re.findall(r'tessera.schedule_hash = "([0-9a-f]{64})"',tile)
    entries = re.findall(r'llvm.func @([\w]+)\(',tile)
    if len(hashes)!=1 or len(entries)!=1:
        raise ValueError('native checkpoint export requires one entry')
    arguments = re.search(r'arguments = (\[[^\]]*\])', text)
    results = re.search(r'\bresults = (\[[^\]]*\])', text)
    bias_policy = re.search(r'bias = (true|false)', text)
    if arguments is None or results is None or bias_policy is None:
        raise ValueError('native checkpoint export lost binding or bias roles')
    names = tuple(json.loads(arguments[1]) + json.loads(results[1]))
    bias = bias_policy[1] == 'true'
    bias_gradient = backward and len(json.loads(results[1])) == 4
    mapping = re.search(r'frontend_argument_indices = array<i64: ([0-9, ]+)>', text)
    if mapping is None and require_frontend_mapping:
        raise ValueError('native AD checkpoint export lost frontend argument mapping')
    indices = tuple(map(int,mapping[1].split(','))) if mapping else ()
    physical_bias = re.search(r'bias_shape = array<i64: ([-0-9, ]+)>', text)
    activity = re.search(r'gradient_activity = array<i64: ([01, ]+)>', text)
    bounds = re.search(r'\bshape_bounds = array<i64: ([0-9, ]+)>',text)
    artifact = ScheduledCheckpointArtifact(graph,schedule,tile,entries[0],hashes[0],backward,names,
        tuple(map(int,shape[1].split(','))),scale_value,causal[1]=='true',bias,bias_gradient,indices,
        tuple(map(int,physical_bias[1].split(','))) if physical_bias else (),
        tuple(map(int,activity[1].split(','))) if activity else (),
        compact_gradients=compact_gradients, compact_launch=compact_launch, compact_threads=compact_threads,
        lse_cotangent="lse_cotangent = true" in text,
        shape_bounds=tuple(map(int,bounds[1].split(","))) if bounds else ())
    artifact.validate()
    return artifact
