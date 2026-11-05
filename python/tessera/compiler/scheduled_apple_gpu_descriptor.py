"""Native Apple GPU runtime-descriptor contracts (E2E-REAL-6, 2026-11-04).

One isolated Apple GPU Graph operation -- ``batched_gemm``, ``transpose``, and
the dynamic-extent ``softmax``, ``gelu``, ``popcount``, ``count_nonzero`` and
``top_k`` -- lowers through the native compiler:
``--tessera-graph-to-schedule`` records a content-addressed contract
(``src/compiler/programming_model/lib/NativeAppleGPUDescriptor.h``),
``--tessera-schedule-to-tile`` emits a Tile value op carrying it, and
``--tile-to-apple_gpu=value=true`` turns that into a verified
``tessera_apple.gpu.kernel_call`` whose runtime entry the lowering derives from
its own table. The package
(``apple_native.package_scheduled_apple_gpu_descriptor``) is projected from the
*replayed* contract, never from the Python Graph object.

The delegate behind every family is a hand-written MPS / MPSGraph / MSL C ABI
entry in ``apple_gpu_runtime.mm``: a Decision #28 Tier-3 implementation reached
through the declared, verified Target IR op -- one lowering path (Decision
#31), not a second authority. The retired Graph-owned constructors
(``apple_native.package_batched_gemm`` and friends) are the declared
differential oracle (Decision #31(a)); production no longer reaches them.

:func:`supports_scheduled_apple_gpu_descriptor` is the host-free admission. It
mirrors what the native contract accepts and normalizes nothing: every
semantic keyword the runtime ABI does not implement is refused (Decision #21a).
"""

from __future__ import annotations

import copy
import functools
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .graph_ir import GraphIRModule

TARGET, ARCHITECTURE = "apple_gpu", "apple7"

#: Graph op -> (family, Tile op suffix, inputs, results). Mirrors
#: ``kAppleGPUDescriptorSpecs`` in ``NativeAppleGPUDescriptor.h``.
APPLE_GPU_DESCRIPTOR_OPS: dict[str, tuple[str, str, int, int]] = {
    "tessera.batched_gemm": ("batched_gemm", "batched_gemm", 2, 1),
    "tessera.transpose": ("transpose", "transpose", 1, 1),
    "tessera.softmax": ("dynamic_softmax", "softmax", 1, 1),
    "tessera.gelu": ("dynamic_gelu", "gelu", 1, 1),
    "tessera.popcount": ("dynamic_popcount", "popcount", 1, 1),
    "tessera.count_nonzero": ("dynamic_count_nonzero", "count_nonzero", 1, 1),
    "tessera.top_k": ("dynamic_topk", "top_k", 1, 2),
}

#: Families whose Graph op is shared with another contract: they belong here
#: only in their dynamic form (static softmax / GELU are scheduled kernels).
DYNAMIC_ONLY = frozenset({"dynamic_softmax", "dynamic_gelu"})

#: (family, storage) -> (C ABI symbol, runtime ABI id, framework). Mirrors the
#: native contract and the Apple GPU value lowering's entry table.
APPLE_GPU_DESCRIPTOR_ENTRIES: dict[tuple[str, str], tuple[str, str, str]] = {
    ("batched_gemm", "fp32"): ("tessera_apple_gpu_bmm_f32", "tessera.apple.bmm.a_b_o_batch_m_n_k.f32.v1", "MPS"),
    ("batched_gemm", "fp16"): ("tessera_apple_gpu_bmm_f16", "tessera.apple.bmm.a_b_o_batch_m_n_k.f16.v1", "MPS"),
    ("batched_gemm", "bf16"): ("tessera_apple_gpu_bmm_bf16", "tessera.apple.bmm.a_b_o_batch_m_n_k.bf16.v1", "MPS"),
    ("transpose", "fp32"): ("tessera_apple_gpu_mpsgraph_transpose_f32",
                            "tessera.apple.mpsgraph.transpose.x_o_dims_perm.f32.v1", "MPSGraph"),
    ("transpose", "fp16"): ("tessera_apple_gpu_mpsgraph_transpose_f16",
                            "tessera.apple.mpsgraph.transpose.x_o_dims_perm.f16.v1", "MPSGraph"),
    ("transpose", "bf16"): ("tessera_apple_gpu_mpsgraph_transpose_f16",
                            "tessera.apple.mpsgraph.transpose.x_o_dims_perm.bf16.v1", "MPSGraph"),
    ("dynamic_softmax", "fp32"): ("tessera_apple_gpu_softmax_f32_status",
                                  "tessera.apple.softmax.x_o_rows_columns.dynamic.f32.v2", "Metal"),
    ("dynamic_gelu", "fp32"): ("tessera_apple_gpu_gelu_f32_status",
                               "tessera.apple.gelu.x_o_elements.dynamic.f32.v2", "Metal"),
    ("dynamic_gelu", "fp16"): ("tessera_apple_gpu_gelu_f16_status",
                               "tessera.apple.gelu.x_o_elements.dynamic.f16.v2", "Metal"),
    ("dynamic_gelu", "bf16"): ("tessera_apple_gpu_gelu_bf16_status",
                               "tessera.apple.gelu.x_o_elements.dynamic.bf16.v2", "Metal"),
    ("dynamic_popcount", "int32"): ("tessera_apple_gpu_popcount_i32",
                                    "tessera.apple.popcount.x_o_elements.dynamic.i32.v1", "Metal"),
    ("dynamic_count_nonzero", "fp32"): (
        "tessera_apple_gpu_count_nonzero_lastaxis_f32",
        "tessera.apple.count_nonzero.x_o_outer_axis_extent.dynamic.f32_i32.v1", "Metal"),
    ("dynamic_topk", "fp32"): (
        "tessera_apple_gpu_topk_f32",
        "tessera.apple.top_k.x_values_indices_rows_columns_k.dynamic.f32_i32.v1", "Metal"),
}

#: Ordered ABI scalars each family's launcher binds: [(name, C type)].
_SCALARS: dict[str, tuple[tuple[str, str], ...]] = {
    "batched_gemm": (), "transpose": (),
    "dynamic_softmax": (("Rows", "int32"), ("Columns", "int32")),
    "dynamic_gelu": (("Elements", "int64"),),
    "dynamic_popcount": (("Elements", "int32"),),
    "dynamic_count_nonzero": (("Outer", "int32"), ("AxisExtent", "int32")),
    "dynamic_topk": (("Rows", "int32"), ("Columns", "int32"), ("K", "int32")),
}

_STORAGE = {"fp32": "f32", "fp16": "f16", "bf16": "bf16", "int32": "i32"}
_DTYPE = {value: key for key, value in _STORAGE.items()}
_INT32_MAX = 2**31 - 1


@dataclass(frozen=True)
class AppleGPUDescriptorRequest:
    """The normalized Graph request one Apple GPU descriptor lowers from."""

    graph_op: str
    kind: str
    input_names: tuple[str, ...]
    output_names: tuple[str, ...]
    shapes: tuple[tuple[int, ...], ...]  # -1 is a dynamic extent
    dtypes: tuple[str, ...]
    semantics: dict[str, Any]


def _shape(raw: Any) -> tuple[int, ...]:
    dims: list[int] = []
    for value in raw:
        if value in ("?", None, -1):
            dims.append(-1)
            continue
        try:
            dim = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("Apple GPU native descriptor extents are integers or '?'") from exc
        if not 0 < dim <= _INT32_MAX:
            raise ValueError("Apple GPU native descriptor requires positive int32 extents")
        dims.append(dim)
    return tuple(dims)


def _owns(op_name: str, shapes: list[tuple[int, ...]]) -> bool:
    spec = APPLE_GPU_DESCRIPTOR_OPS.get(op_name)
    if spec is None:
        return False
    return spec[0] not in DYNAMIC_ONLY or any(-1 in shape for shape in shapes[:spec[2]])


def owns(module: GraphIRModule) -> bool:
    """Whether this contract is ``module``'s only native authority.

    True for every isolated request on an owned op, admitted or not; a static
    softmax/GELU belongs to the shared semantic-kernel contract instead.
    """
    if len(module.functions) != 1 or len(module.functions[0].body) != 1:
        return False
    function = module.functions[0]
    op = function.body[0]
    args = {arg.name: arg for arg in function.args}
    try:
        shapes = [_shape(args[name.removeprefix("%")].ir_type.shape) for name in op.operands]
    except (KeyError, ValueError):
        return op.op_name in APPLE_GPU_DESCRIPTOR_OPS and \
            APPLE_GPU_DESCRIPTOR_OPS[op.op_name][0] not in DYNAMIC_ONLY
    return _owns(op.op_name, shapes)


def _last_axis(kwargs: dict[str, Any]) -> bool:
    axis = kwargs.get("axis", -1)
    return not isinstance(axis, bool) and axis in {-1, 1}


def admit(module: GraphIRModule) -> AppleGPUDescriptorRequest:
    """Admit one Apple GPU native descriptor request or raise ``ValueError``."""
    if len(module.functions) != 1 or len(module.functions[0].body) != 1:
        raise ValueError("Apple GPU native descriptor requires one Graph function and operation")
    function = module.functions[0]
    op = function.body[0]
    spec = APPLE_GPU_DESCRIPTOR_OPS.get(op.op_name)
    if spec is None:
        raise ValueError(f"{op.op_name} has no Apple GPU native descriptor contract")
    kind, _tile, n_inputs, n_results = spec
    args = {arg.name: arg for arg in function.args}
    inputs = tuple(value.removeprefix("%") for value in op.operands)
    outputs = tuple(value.removeprefix("%") for value in function.return_values)
    if (len(inputs) != n_inputs or len(set(inputs)) != n_inputs or set(inputs) != set(args)
            or len(function.args) != n_inputs):
        raise ValueError("Apple GPU native descriptor operands must be its distinct entry arguments")
    if (len(outputs) != n_results or len(function.result_types) != n_results
            or len(set(inputs + outputs)) != n_inputs + n_results or "" in outputs
            or any(not name.isidentifier() for name in inputs + outputs)):
        raise ValueError("Apple GPU native descriptor binds each result to a distinct name")
    shapes = [_shape(args[name].ir_type.shape) for name in inputs]
    dtypes = [str(args[name].ir_type.dtype) for name in inputs]
    shapes += [_shape(result.shape) for result in function.result_types]
    dtypes += [str(result.dtype) for result in function.result_types]
    if not _owns(op.op_name, shapes):
        raise ValueError(f"static {op.op_name} lowers through the scheduled semantic kernel")
    if any(dtype not in _STORAGE for dtype in dtypes):
        raise ValueError("Apple GPU native descriptor storage is f32/f16/bf16/i32 only")
    raw = {key: value for key, value in op.kwargs.items() if value is not None}
    storage = dtypes[0]
    floats = storage in {"fp32", "fp16", "bf16"}
    semantics: dict[str, Any] = {}

    def refuse_unknown(*known: str) -> None:
        unknown = sorted(set(raw) - set(known))
        if unknown:
            raise ValueError(f"Apple GPU native {kind} has unsupported keyword(s) {unknown}")

    def dynamic(shape: tuple[int, ...], rank: int) -> bool:
        return len(shape) == rank and all(dim == -1 for dim in shape)

    if kind == "batched_gemm":
        refuse_unknown()
        lhs, rhs, out = shapes
        if (any(-1 in shape for shape in shapes) or len(lhs) != 3 or len(rhs) != 3
                or not floats or set(dtypes) != {storage}):
            raise ValueError("Apple GPU batched_gemm requires static rank-3 f32/f16/bf16 of one storage")
        batch, m, k = lhs
        n = rhs[2]
        if rhs != (batch, k, n) or out != (batch, m, n):
            raise ValueError("Apple GPU batched_gemm requires [B,M,K] x [B,K,N] -> [B,M,N] with no broadcasting")
        semantics["broadcast_b"] = False
    elif kind == "transpose":
        refuse_unknown("axes")
        x, out = shapes
        rank = len(x)
        if -1 in x or -1 in out or rank not in {2, 3} or not floats or dtypes[1] != storage:
            raise ValueError("Apple GPU transpose requires a static rank-2/rank-3 f32/f16/bf16 permutation")
        raw_axes = raw.get("axes", tuple(range(rank - 1, -1, -1)))
        if not isinstance(raw_axes, (list, tuple)) or any(
                isinstance(a, bool) or not isinstance(a, int) for a in raw_axes):
            raise ValueError("Apple GPU transpose axes must be integers")
        axes = tuple(int(a) % rank if -rank <= int(a) < rank else -1 for a in raw_axes)
        if len(axes) != rank or set(axes) != set(range(rank)):
            raise ValueError("Apple GPU transpose permutation must cover the operand rank")
        if out != tuple(x[a] for a in axes):
            raise ValueError("Apple GPU transpose result shape disagrees with its permutation")
        semantics["axes"] = axes
    elif kind == "dynamic_softmax":
        refuse_unknown("axis")
        if not dynamic(shapes[0], 2) or shapes[1] != shapes[0] or dtypes != ["fp32", "fp32"] \
                or not _last_axis(raw):
            raise ValueError("Apple GPU dynamic softmax requires ?x? f32 last-axis input and output")
        semantics["axis"] = -1
    elif kind == "dynamic_gelu":
        refuse_unknown()
        if not dynamic(shapes[0], 2) or shapes[1] != shapes[0] or not floats or dtypes[1] != storage:
            raise ValueError("Apple GPU dynamic GELU requires matching ?x? f32/f16/bf16 tensors")
        semantics["approximation"] = "tanh"
    elif kind == "dynamic_popcount":
        refuse_unknown()
        if not dynamic(shapes[0], 1) or shapes[1] != shapes[0] or dtypes != ["int32", "int32"]:
            raise ValueError("Apple GPU dynamic popcount requires matching rank-1 ? i32 tensors")
    elif kind == "dynamic_count_nonzero":
        refuse_unknown("axis", "keepdims")
        keepdims = raw.get("keepdims", False)
        if (not dynamic(shapes[0], 2) or not dynamic(shapes[1], 1) or dtypes != ["fp32", "int32"]
                or not _last_axis(raw) or keepdims is not False):
            raise ValueError("Apple GPU dynamic count_nonzero requires ?x? f32 -> ? i32 over the "
                             "last axis with keepdims = False")
        semantics.update(axis=-1, keepdims=False)
    else:  # dynamic_topk
        refuse_unknown("k", "axis")
        top_k = raw.get("k")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or not 0 < top_k <= _INT32_MAX:
            raise ValueError("Apple GPU dynamic top_k requires a static positive integer K")
        if (not dynamic(shapes[0], 2) or shapes[1] != (-1, top_k) or shapes[2] != (-1, top_k)
                or dtypes != ["fp32", "fp32", "int32"] or not _last_axis(raw)):
            raise ValueError("Apple GPU dynamic top_k requires ?x? f32 -> (?xK f32, ?xK i32) "
                             "over the last axis")
        semantics.update(axis=-1, k=top_k, value_order="descending", nan_policy="last",
                         tie_policy="lower_index")
    if (kind, storage) not in APPLE_GPU_DESCRIPTOR_ENTRIES:
        raise ValueError(f"Apple GPU native {kind} has no runtime entry for {storage}")
    return AppleGPUDescriptorRequest(
        graph_op=op.op_name, kind=kind, input_names=inputs, output_names=outputs,
        shapes=tuple(shapes), dtypes=tuple(dtypes), semantics=semantics,
    )


def supports_scheduled_apple_gpu_descriptor(module: GraphIRModule, *, target: str = TARGET) -> bool:
    """Whether the native Apple GPU descriptor contract admits ``module``."""
    if target != TARGET:
        return False
    try:
        admit(module)
    except ValueError:
        return False
    return True


@functools.cache
def apple_gpu_value_lowering_available(tool: str) -> bool:
    """Whether ``tool`` registers this contract and the Apple GPU value lowering.

    A ``tessera-opt`` built without the Apple backend parses Graph IR but cannot
    produce ``tessera_apple.gpu.kernel_call``, so a family owned here is not
    packageable with it. Probed once per binary rather than assumed.
    """
    probe = ('module attributes {tessera.target = "apple_gpu", tessera.arch = "apple7", '
             'tessera.launch_bindings = ["x", "y"]} {\n'
             '  func.func @f(%x: tensor<?xi32>) -> tensor<?xi32> {\n'
             '    %0 = tessera.popcount %x : (tensor<?xi32>) -> tensor<?xi32>\n'
             '    return %0 : tensor<?xi32>\n  }\n}\n')
    try:
        _schedule, _tile, target_ir = _boundaries(Path(tool), probe)
    except (OSError, RuntimeError):
        return False
    return "tessera_apple.gpu.kernel_call" in target_ir


def compiler_available() -> bool:
    from .scheduled_matmul import find_tessera_opt

    tool = find_tessera_opt()
    return tool is not None and apple_gpu_value_lowering_available(str(tool))


def _require_compiler() -> Path:
    from .scheduled_matmul import find_tessera_opt

    tool = find_tessera_opt()
    if tool is None or not apple_gpu_value_lowering_available(str(tool)):
        raise RuntimeError(
            "Apple GPU native descriptor requires production tessera-opt with the Apple GPU "
            "descriptor contract and value lowering (--tile-to-apple_gpu=value=true)"
        )
    return tool


@dataclass(frozen=True)
class ScheduledAppleGPUDescriptorArtifact:
    """One replayable Graph -> Schedule -> Tile -> Target Apple GPU descriptor."""

    target: str
    architecture: str
    kind: str
    graph_op: str
    input_names: tuple[str, ...]
    output_names: tuple[str, ...]
    shapes: tuple[tuple[int, ...], ...]
    dtypes: tuple[str, ...]
    semantics: dict[str, Any]
    graph_ir: str
    schedule_ir: str
    tile_ir: str
    target_ir: str
    schedule_digest: str

    def validate(self) -> None:
        if (self.target, self.architecture) != (TARGET, ARCHITECTURE):
            raise ValueError("Apple GPU native descriptor artifact names another target")
        spec = APPLE_GPU_DESCRIPTOR_OPS.get(self.graph_op)
        if spec is None or spec[0] != self.kind:
            raise ValueError("Apple GPU native descriptor artifact kind disagrees with its op")
        if (len(self.input_names) != spec[2] or len(self.output_names) != spec[3]
                or len(self.shapes) != spec[2] + spec[3] or len(self.dtypes) != len(self.shapes)):
            raise ValueError("Apple GPU native descriptor artifact arity disagrees with its op")
        if re.fullmatch(r"[0-9a-f]{64}", self.schedule_digest) is None:
            raise ValueError("Apple GPU native descriptor artifact requires a schedule digest")


def _boundaries(tool: Any, graph_ir: str) -> tuple[str, str, str]:
    from .scheduled_matmul import run_tessera_opt

    schedule_ir = run_tessera_opt(tool, graph_ir, "--tessera-graph-to-schedule")
    tile_ir = run_tessera_opt(tool, schedule_ir, "--tessera-schedule-to-tile")
    target_ir = run_tessera_opt(tool, tile_ir, "--tile-to-apple_gpu=value=true")
    return schedule_ir, tile_ir, target_ir


def lower_scheduled_apple_gpu_descriptor(
    module: GraphIRModule, *, target: str = TARGET,
) -> ScheduledAppleGPUDescriptorArtifact:
    """Graph -> Schedule -> Tile -> Target for one admitted request."""
    if target != TARGET:
        raise ValueError("Apple GPU native descriptor lowers for apple_gpu only")
    request = admit(module)
    tool = _require_compiler()
    targeted = copy.deepcopy(module)
    targeted.module_attrs["tessera.target"] = f'"{TARGET}"'
    targeted.module_attrs["tessera.arch"] = f'"{ARCHITECTURE}"'
    targeted.module_attrs["tessera.launch_bindings"] = json.dumps(
        list(request.input_names + request.output_names))
    graph_ir = targeted.to_mlir(target=TARGET, canonical=True)
    schedule_ir, tile_ir, target_ir = _boundaries(tool, graph_ir)
    digests = re.findall(r'schedule\.artifact .*?hash = "([0-9a-f]{64})"', schedule_ir)
    if len(digests) != 1:
        raise ValueError("Apple GPU native descriptor Schedule requires exactly one artifact record")
    return ScheduledAppleGPUDescriptorArtifact(
        target=TARGET, architecture=ARCHITECTURE, kind=request.kind, graph_op=request.graph_op,
        input_names=request.input_names, output_names=request.output_names,
        shapes=request.shapes, dtypes=request.dtypes, semantics=dict(request.semantics),
        graph_ir=graph_ir, schedule_ir=schedule_ir, tile_ir=tile_ir, target_ir=target_ir,
        schedule_digest=digests[0],
    )


@dataclass(frozen=True)
class AppleGPUDescriptorProjection:
    kind: str
    tile_op: str
    symbol: str
    abi: str
    framework: str
    bindings: tuple[str, ...]
    inputs: int
    shapes: tuple[tuple[int, ...], ...]
    dtypes: tuple[str, ...]
    scalars: tuple[tuple[str, str], ...]
    semantics: dict[str, Any]
    schedule_digest: str


def _line_ops(ir: str, dialect: str) -> list[str]:
    """Operation names of ``dialect`` that begin a line (generic form included)."""
    return re.findall(r'(?m)^\s*(?:%\S+ = )?"?(' + dialect + r'\.[A-Za-z_.0-9]+)', ir)


def _normalized(value: Any) -> Any:
    return list(value) if isinstance(value, tuple) else value


def verify_apple_gpu_descriptor_projection(
    artifact: ScheduledAppleGPUDescriptorArtifact,
) -> AppleGPUDescriptorProjection:
    """Replay every boundary and project the serialized contract.

    Each stored stage must be byte-equal to the production compiler's replay of
    the stage before it, and every descriptor field the artifact carries must
    equal the contract the native Schedule serialized. The Target IR must hold
    exactly one verified ``tessera_apple.gpu.kernel_call`` naming the
    contract's symbol, ABI, framework and schedule digest.
    """
    from .native_x86_kernel import parse_contract

    artifact.validate()
    tool = _require_compiler()
    schedule_ir, tile_ir, target_ir = _boundaries(tool, artifact.graph_ir)
    if schedule_ir != artifact.schedule_ir:
        raise ValueError("Apple GPU native descriptor Schedule disagrees with Graph replay")
    if tile_ir != artifact.tile_ir:
        raise ValueError("Apple GPU native descriptor Tile disagrees with Schedule replay")
    if target_ir != artifact.target_ir:
        raise ValueError("Apple GPU native descriptor Target IR disagrees with Tile replay")
    header = re.match(r'\s*module attributes \{([^{}]*)\}', artifact.tile_ir)
    if header is None or any(
        re.search(r'(?:^|,)\s*' + re.escape(key) + r' = "' + value + r'"(?:,|$)', header[1]) is None
        for key, value in (("tessera.target", TARGET), ("tessera.arch", ARCHITECTURE))
    ):
        raise ValueError("Apple GPU native descriptor parent target disagrees")
    contract = parse_contract(artifact.tile_ir, "apple_gpu_descriptor")
    if parse_contract(artifact.target_ir, "apple_gpu_descriptor") != contract:
        raise ValueError("Apple GPU native descriptor Target contract disagrees with Tile")
    if (re.findall(r'tessera\.schedule_hash = "([0-9a-f]{64})"', artifact.tile_ir)
            != [artifact.schedule_digest]):
        raise ValueError("Apple GPU native descriptor Tile artifact has a stale schedule digest")
    tile_op = "tile." + APPLE_GPU_DESCRIPTOR_OPS[artifact.graph_op][1]
    if (_line_ops(artifact.tile_ir, "tile") != [tile_op]
            or _line_ops(artifact.tile_ir, "tessera")):
        raise ValueError("Apple GPU native descriptor Tile artifact requires exactly its one value op")
    bindings = contract.get("bindings")
    shapes = tuple(tuple(shape) for shape in contract.get("shapes", ()))
    dtypes = tuple(_DTYPE.get(s, "") for s in contract.get("storage", ()))
    semantics = contract.get("semantics")
    raw_scalars = contract.get("scalars")
    if (not isinstance(bindings, list) or not isinstance(semantics, dict)
            or not isinstance(raw_scalars, list) or contract.get("layout") != "row_major"
            or any(not isinstance(s, list) or len(s) != 2 for s in raw_scalars)):
        raise ValueError("Apple GPU native descriptor contract is malformed")
    scalars = tuple((str(s[0]), str(s[1])) for s in raw_scalars)
    expected_semantics = {key: _normalized(value) for key, value in artifact.semantics.items()}
    actual_semantics = {key: _normalized(value) for key, value in semantics.items()}
    if (contract.get("kind") != artifact.kind or contract.get("graph_op") != artifact.graph_op
            or contract.get("tile_op") != tile_op
            or tuple(bindings) != artifact.input_names + artifact.output_names
            or contract.get("inputs") != len(artifact.input_names)
            or shapes != artifact.shapes or dtypes != artifact.dtypes
            or actual_semantics != expected_semantics
            or scalars != _SCALARS[artifact.kind]):
        raise ValueError("Apple GPU native descriptor fields disagree with native IR")
    entry = APPLE_GPU_DESCRIPTOR_ENTRIES.get((artifact.kind, artifact.dtypes[0]))
    symbol, abi, framework = (str(contract.get(k, "")) for k in ("symbol", "abi", "framework"))
    if entry != (symbol, abi, framework):
        raise ValueError("Apple GPU native descriptor runtime entry disagrees with the contract")
    calls = re.findall(r"(?m)^\s*(?:%\S+ = )?(tessera_apple\.[A-Za-z_.0-9]+)(.*)$", artifact.target_ir)
    if (len(calls) != 1 or calls[0][0] != "tessera_apple.gpu.kernel_call"
            or _line_ops(artifact.target_ir, "tile") or _line_ops(artifact.target_ir, "tessera")):
        raise ValueError("Apple GPU native descriptor Target IR requires exactly one "
                         "tessera_apple.gpu.kernel_call")
    attrs = calls[0][1]
    op_kind = tile_op.removeprefix("tile.")
    expected = {"op_kind": f'"{op_kind}"', "symbol": f'"{symbol}"', "abi": f'"{abi}"',
                "status": '"executable"', "framework": f'"{framework}"',
                "tessera.schedule_hash": f'"{artifact.schedule_digest}"'}
    for key, value in expected.items():
        if re.search(r"[{,]\s*" + re.escape(key) + r" = " + re.escape(value) + r"[,}]", attrs) is None:
            raise ValueError(f"Apple GPU native descriptor Target IR disagrees on {key}")
    return AppleGPUDescriptorProjection(
        kind=artifact.kind, tile_op=tile_op, symbol=symbol, abi=abi, framework=framework,
        bindings=tuple(bindings), inputs=len(artifact.input_names), shapes=shapes,
        dtypes=dtypes, scalars=scalars, semantics=dict(semantics),
        schedule_digest=artifact.schedule_digest,
    )


__all__ = [
    "APPLE_GPU_DESCRIPTOR_ENTRIES", "APPLE_GPU_DESCRIPTOR_OPS", "AppleGPUDescriptorProjection",
    "AppleGPUDescriptorRequest", "DYNAMIC_ONLY", "ScheduledAppleGPUDescriptorArtifact", "admit",
    "apple_gpu_value_lowering_available", "compiler_available",
    "lower_scheduled_apple_gpu_descriptor", "owns", "supports_scheduled_apple_gpu_descriptor",
    "verify_apple_gpu_descriptor_projection",
]
