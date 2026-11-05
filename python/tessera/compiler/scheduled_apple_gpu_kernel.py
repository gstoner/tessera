"""Native Apple GPU linalg contracts (E2E-REAL-6, 2026-11-04).

One isolated static Apple GPU Graph operation -- ``cholesky``, ``tri_solve``,
``cholesky_solve`` or reduced ``svd`` -- lowers through the native compiler:
``--tessera-graph-to-schedule`` records a content-addressed contract
(``src/compiler/programming_model/lib/NativeAppleGPUKernel.h``),
``--tessera-schedule-to-tile`` emits the registered Tile value op, and
``--tile-to-apple_gpu=value=true`` turns it into a verified
``tessera_apple.gpu.kernel_call``. The package
(``apple_native.package_scheduled_apple_gpu_kernel``) is projected from the
*replayed* contract, never from the Python Graph object.

The delegate behind every family is a hand-written MSL / MPS C ABI entry in
``apple_gpu_runtime.mm``: a Decision #28 Tier-3 implementation reached through
the declared, verified Target IR op, which is one lowering path (Decision #31),
not a second authority. The retired Graph-owned constructor is the declared
differential oracle in ``tests/_support/apple_gpu_linalg_baseline.py``
(Decision #31(a)).

:func:`supports_scheduled_apple_gpu_kernel` is the host-free admission. It
mirrors what the native contract accepts and normalizes nothing: every semantic
keyword the Metal kernels do not implement is refused (Decision #21a).
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

#: Graph op -> (kind, inputs, results). Mirrors ``kAppleGPUKernelSpecs``.
APPLE_GPU_KERNEL_OPS: dict[str, tuple[str, int, int]] = {
    "tessera.cholesky": ("cholesky", 1, 1),
    "tessera.tri_solve": ("tri_solve", 2, 1),
    "tessera.cholesky_solve": ("cholesky_solve", 2, 1),
    "tessera.svd": ("svd", 1, 3),
}

#: (kind, batched) -> (C ABI symbol, ABI id). Mirrors ``appleGPUKernelEntry``
#: and the value lowering's ``kNativeAppleGPUKernelEntries``.
APPLE_GPU_KERNEL_ENTRIES: dict[tuple[str, bool], tuple[str, str]] = {
    **{(kind, batched): (symbol, f"tessera.apple.value.{kind}.f32.v1")
       for kind, symbol in (("cholesky", "tessera_apple_gpu_cholesky_f32"),
                            ("tri_solve", "tessera_apple_gpu_tri_solve_f32"),
                            ("cholesky_solve", "tessera_apple_gpu_solve_cholesky_f32"))
       for batched in (False, True)},
    ("svd", False): ("tessera_apple_gpu_svd_reduced_f32", "tessera.apple.svd.a_u_s_vh_m_n.f32.v1"),
    ("svd", True): ("tessera_apple_gpu_svd_reduced_batched_f32",
                    "tessera.apple.svd.a_u_s_vh_batch_m_n.f32.v1"),
}

#: The semantic flags each family materializes, with their ODS defaults.
_FLAGS: dict[str, dict[str, bool]] = {
    "cholesky": {"lower": True},
    "tri_solve": {"lower": True, "trans": False, "unit_diag": False},
    "cholesky_solve": {"lower": True},
    "svd": {"full_matrices": False},
}

_INT32_MAX = 2**31 - 1
TARGET = ARCHITECTURE = "apple_gpu"


@dataclass(frozen=True)
class AppleGPUKernelRequest:
    """The normalized Graph request one Apple GPU native kernel lowers from."""

    graph_op: str
    kind: str
    symbol: str
    abi: str
    input_names: tuple[str, ...]
    output_names: tuple[str, ...]
    shapes: tuple[tuple[int, ...], ...]
    semantics: dict[str, bool]


def _static(shape: Any) -> tuple[int, ...]:
    try:
        dims = tuple(int(value) for value in shape)
    except (TypeError, ValueError) as exc:
        raise ValueError("Apple GPU native kernel requires static shapes") from exc
    if not dims or any(value <= 0 for value in dims):
        raise ValueError("Apple GPU native kernel requires positive extents")
    count = 1
    for value in dims:
        count *= value
    if count > _INT32_MAX:
        raise ValueError("Apple GPU native kernel tensor exceeds the int32 ABI")
    return dims


def admit(module: GraphIRModule) -> AppleGPUKernelRequest:
    """Admit one Apple GPU native kernel request or raise ``ValueError``."""
    if len(module.functions) != 1 or len(module.functions[0].body) != 1:
        raise ValueError("Apple GPU native kernel requires one Graph function and operation")
    function = module.functions[0]
    op = function.body[0]
    spec = APPLE_GPU_KERNEL_OPS.get(op.op_name)
    if spec is None:
        raise ValueError(f"{op.op_name} has no Apple GPU native kernel contract")
    kind, n_inputs, n_results = spec
    args = {arg.name: arg for arg in function.args}
    inputs = tuple(value.removeprefix("%") for value in op.operands)
    outputs = tuple(value.removeprefix("%") for value in function.return_values)
    if (len(inputs) != n_inputs or len(set(inputs)) != n_inputs or set(inputs) != set(args)
            or len(function.args) != n_inputs):
        raise ValueError("Apple GPU native kernel operands must be its distinct entry arguments")
    if (len(outputs) != n_results or len(function.result_types) != n_results
            or len(set(inputs + outputs)) != n_inputs + n_results or "" in outputs):
        raise ValueError("Apple GPU native kernel binds each result to a distinct name")
    shapes = [_static(args[name].ir_type.shape) for name in inputs]
    dtypes = [str(args[name].ir_type.dtype) for name in inputs]
    shapes += [_static(result.shape) for result in function.result_types]
    dtypes += [str(result.dtype) for result in function.result_types]
    if any(dtype != "fp32" for dtype in dtypes):
        raise ValueError("Apple GPU native kernel storage is f32 only")
    # None-valued keywords mean "not given" (the frontend's spelling).
    raw = {key: value for key, value in op.kwargs.items() if value is not None}
    unknown = sorted(set(raw) - set(_FLAGS[kind]))
    if unknown:
        raise ValueError(f"Apple GPU native {kind} has unsupported keyword(s) {unknown}")
    semantics: dict[str, bool] = {}
    for name, default in _FLAGS[kind].items():
        value = raw.get(name, default)
        if not isinstance(value, bool):
            raise ValueError(f"Apple GPU native {kind} {name} must be a boolean")
        semantics[name] = value

    matrix = shapes[0]
    if len(matrix) not in {2, 3}:
        raise ValueError(f"Apple GPU {kind} requires a rank-2 matrix or a rank-3 batch")
    batched = len(matrix) == 3
    lead = matrix[:-2]
    if kind == "svd":
        if semantics["full_matrices"]:
            raise ValueError("Apple GPU svd implements full_matrices = False only")
        m, n = matrix[-2:]
        if m < n:
            raise ValueError("Apple GPU svd requires a tall matrix (M >= N)")
        if shapes[1:] != [lead + (m, n), lead + (n,), lead + (n, n)]:
            raise ValueError("Apple GPU svd requires reduced (U [..,M,N], S [..,N], Vh [..,N,N])")
    else:
        if matrix[-1] != matrix[-2]:
            raise ValueError(f"Apple GPU {kind} requires a square [N,N] or [B,N,N] matrix")
        if kind in {"cholesky", "cholesky_solve"} and not semantics["lower"]:
            raise ValueError(f"Apple GPU {kind} implements lower = True only")
        if kind == "cholesky":
            if shapes[1] != matrix:
                raise ValueError("Apple GPU cholesky writes a factor of the matrix shape")
        else:
            rhs, out = shapes[1], shapes[2]
            if len(rhs) != len(matrix) or rhs[:-1] != matrix[:-1] or out != rhs:
                raise ValueError(f"Apple GPU {kind} requires rhs [..,N,K] with the matrix's "
                                 "batch and a result of the rhs shape")
    symbol, abi = APPLE_GPU_KERNEL_ENTRIES[(kind, batched)]
    return AppleGPUKernelRequest(
        graph_op=op.op_name, kind=kind, symbol=symbol, abi=abi, input_names=inputs,
        output_names=outputs, shapes=tuple(shapes), semantics=semantics,
    )


def supports_scheduled_apple_gpu_kernel(module: GraphIRModule, *, target: str = TARGET) -> bool:
    """Whether the native Apple GPU kernel contract admits ``module``."""
    if target != TARGET:
        return False
    try:
        admit(module)
    except ValueError:
        return False
    return True


@functools.cache
def apple_gpu_value_lowering_available(tool: str) -> bool:
    """Whether ``tool`` registers the native Apple GPU value lowering.

    A ``tessera-opt`` built without the Apple backend (or from before this
    slice) parses Graph IR but cannot produce ``tessera_apple.gpu.kernel_call``
    from the native contract, so these families are not packageable with it.
    Probed once per binary rather than assumed.
    """
    probe = ('module attributes {tessera.target = "apple_gpu", tessera.arch = "apple_gpu", '
             'tessera.launch_bindings = ["a", "l"]} {\n'
             '  func.func @f(%a: tensor<1x1xf32>) -> tensor<1x1xf32> {\n'
             '    %0 = tessera.cholesky %a : (tensor<1x1xf32>) -> tensor<1x1xf32>\n'
             '    return %0 : tensor<1x1xf32>\n  }\n}\n')
    try:
        _, _, target_ir = _boundaries(Path(tool), probe)
    except (OSError, RuntimeError, ValueError):
        return False
    return "tessera_apple.gpu.kernel_call" in target_ir


def lowering_available() -> bool:
    """Whether this host's production compiler can produce the native route."""
    from .scheduled_matmul import find_tessera_opt

    tool = find_tessera_opt()
    return tool is not None and apple_gpu_value_lowering_available(str(tool))


@dataclass(frozen=True)
class ScheduledAppleGPUKernelArtifact:
    """One replayable Graph -> Schedule -> Tile -> Target Apple GPU kernel."""

    target: str
    architecture: str
    kind: str
    graph_op: str
    symbol: str
    input_names: tuple[str, ...]
    output_names: tuple[str, ...]
    shapes: tuple[tuple[int, ...], ...]
    semantics: dict[str, bool]
    graph_ir: str
    schedule_ir: str
    tile_ir: str
    target_ir: str
    schedule_digest: str

    def validate(self) -> None:
        if (self.target, self.architecture) != (TARGET, ARCHITECTURE):
            raise ValueError("Apple GPU native kernel artifact names another target")
        spec = APPLE_GPU_KERNEL_OPS.get(self.graph_op)
        if spec is None or spec[0] != self.kind:
            raise ValueError("Apple GPU native kernel artifact kind disagrees with its op")
        if (len(self.input_names) != spec[1] or len(self.output_names) != spec[2]
                or len(self.shapes) != spec[1] + spec[2]):
            raise ValueError("Apple GPU native kernel artifact arity disagrees with its op")
        if not self.shapes or len(self.shapes[0]) not in {2, 3}:
            raise ValueError("Apple GPU native kernel artifact requires a rank-2 or rank-3 matrix")
        if APPLE_GPU_KERNEL_ENTRIES[(self.kind, len(self.shapes[0]) == 3)][0] != self.symbol:
            raise ValueError("Apple GPU native kernel artifact symbol disagrees with its op")
        if re.fullmatch(r"[0-9a-f]{64}", self.schedule_digest) is None:
            raise ValueError("Apple GPU native kernel artifact requires a schedule digest")


def _boundaries(tool: Any, graph_ir: str) -> tuple[str, str, str]:
    from .scheduled_matmul import run_tessera_opt

    schedule_ir = run_tessera_opt(tool, graph_ir, "--tessera-graph-to-schedule")
    tile_ir = run_tessera_opt(tool, schedule_ir, "--tessera-schedule-to-tile")
    target_ir = run_tessera_opt(tool, tile_ir, "--tile-to-apple_gpu=value=true")
    return schedule_ir, tile_ir, target_ir


def _require_tool() -> Path:
    from .scheduled_matmul import find_tessera_opt

    tool = find_tessera_opt()
    if tool is None:
        raise RuntimeError("Apple GPU native kernel requires production tessera-opt")
    if not apple_gpu_value_lowering_available(str(tool)):
        raise RuntimeError(
            f"Apple GPU native kernel: tessera-opt at {tool} does not register the native "
            "Apple GPU value lowering (--tile-to-apple_gpu=value=true); rebuild it with "
            "the Apple backend")
    return Path(tool)


def lower_scheduled_apple_gpu_kernel(
    module: GraphIRModule, *, target: str = TARGET,
) -> ScheduledAppleGPUKernelArtifact:
    """Graph -> Schedule -> Tile -> Target for one admitted request."""
    if target != TARGET:
        raise ValueError("Apple GPU native kernel lowers for apple_gpu only")
    request = admit(module)
    tool = _require_tool()
    targeted = copy.deepcopy(module)
    targeted.module_attrs["tessera.target"] = f'"{TARGET}"'
    targeted.module_attrs["tessera.arch"] = f'"{ARCHITECTURE}"'
    targeted.module_attrs["tessera.launch_bindings"] = json.dumps(
        list(request.input_names + request.output_names))
    graph_ir = targeted.to_mlir(target=TARGET, canonical=True)
    schedule_ir, tile_ir, target_ir = _boundaries(tool, graph_ir)
    digests = re.findall(r'schedule\.artifact .*?hash = "([0-9a-f]{64})"', schedule_ir)
    if len(digests) != 1:
        raise ValueError("Apple GPU native kernel Schedule requires exactly one artifact record")
    return ScheduledAppleGPUKernelArtifact(
        target=TARGET, architecture=ARCHITECTURE, kind=request.kind, graph_op=request.graph_op,
        symbol=request.symbol, input_names=request.input_names, output_names=request.output_names,
        shapes=request.shapes, semantics=dict(request.semantics),
        graph_ir=graph_ir, schedule_ir=schedule_ir, tile_ir=tile_ir, target_ir=target_ir,
        schedule_digest=digests[0],
    )


@dataclass(frozen=True)
class AppleGPUKernelProjection:
    kind: str
    symbol: str
    abi: str
    bindings: tuple[str, ...]
    inputs: int
    shapes: tuple[tuple[int, ...], ...]
    scalars: dict[str, int]
    semantics: dict[str, bool]
    schedule_digest: str


_FLAG_NAMES = frozenset({"lower", "trans", "unit_diag", "full_matrices"})


def verify_apple_gpu_kernel_projection(artifact: ScheduledAppleGPUKernelArtifact) -> AppleGPUKernelProjection:
    """Replay every boundary and project the serialized contract.

    Each stored stage must be byte-equal to the production compiler's replay of
    the stage before it, and every descriptor field the artifact carries must
    equal the contract the native Schedule serialized. The Target IR must hold
    exactly one verified ``tessera_apple.gpu.kernel_call`` naming the
    contract's symbol, ABI, semantic flags and schedule digest. Every check
    runs before anything reaches the runtime.
    """
    from .native_x86_kernel import parse_contract

    artifact.validate()
    tool = _require_tool()
    schedule_ir, tile_ir, target_ir = _boundaries(tool, artifact.graph_ir)
    if schedule_ir != artifact.schedule_ir:
        raise ValueError("Apple GPU native kernel Schedule disagrees with Graph replay")
    if tile_ir != artifact.tile_ir:
        raise ValueError("Apple GPU native kernel Tile disagrees with Schedule replay")
    if target_ir != artifact.target_ir:
        raise ValueError("Apple GPU native kernel Target IR disagrees with Tile replay")
    header = re.match(r'\s*module attributes \{([^{}]*)\}', artifact.tile_ir)
    if header is None or any(
        re.search(r'(?:^|,)\s*' + re.escape(key) + r' = "' + value + r'"(?:,|$)', header[1]) is None
        for key, value in (("tessera.target", TARGET), ("tessera.arch", ARCHITECTURE))
    ):
        raise ValueError("Apple GPU native kernel parent target disagrees")
    contract = parse_contract(artifact.tile_ir, "apple_gpu_kernel")
    if re.findall(r'tessera\.schedule_hash = "([0-9a-f]{64})"', artifact.tile_ir) != [artifact.schedule_digest]:
        raise ValueError("Apple GPU native kernel Tile artifact has a stale schedule digest")
    tile_ops = re.findall(r"(?m)^\s*(?:%\S+ = )?(tile\.[A-Za-z_0-9]+)", artifact.tile_ir)
    if tile_ops != [f"tile.{artifact.kind}"] or "tessera." + artifact.kind + " " in artifact.tile_ir:
        raise ValueError("Apple GPU native kernel Tile artifact requires exactly its one value op")
    bindings = contract.get("bindings")
    shapes = tuple(tuple(shape) for shape in contract.get("shapes", ()))
    semantics = contract.get("semantics")
    scalars = contract.get("scalars")
    abi = str(contract.get("abi", ""))
    if (not isinstance(bindings, list) or not isinstance(semantics, dict)
            or not isinstance(scalars, dict) or contract.get("layout") != "row_major"
            or contract.get("framework") != "Metal"
            or tuple(contract.get("storage", ())) != ("f32",) * len(shapes)):
        raise ValueError("Apple GPU native kernel contract is malformed")
    batched = len(artifact.shapes[0]) == 3
    if (contract.get("kind") != artifact.kind or contract.get("graph_op") != artifact.graph_op
            or contract.get("symbol") != artifact.symbol
            or (artifact.symbol, abi) != APPLE_GPU_KERNEL_ENTRIES[(artifact.kind, batched)]
            or tuple(bindings) != artifact.input_names + artifact.output_names
            or contract.get("inputs") != len(artifact.input_names)
            or shapes != artifact.shapes or semantics != artifact.semantics):
        raise ValueError("Apple GPU native kernel descriptor fields disagree with native IR")
    calls = re.findall(r"(?m)^\s*(?:%\S+ = )?(tessera_apple\.[A-Za-z_.0-9]+)(.*)$", artifact.target_ir)
    if len(calls) != 1 or calls[0][0] != "tessera_apple.gpu.kernel_call" or "tile." in artifact.target_ir:
        raise ValueError("Apple GPU native kernel Target IR requires exactly one tessera_apple.gpu.kernel_call")
    attrs = calls[0][1]
    expected = {"op_kind": f'"{artifact.kind}"', "symbol": f'"{artifact.symbol}"', "abi": f'"{abi}"',
                "status": '"executable"', "framework": '"Metal"',
                "tessera.schedule_hash": f'"{artifact.schedule_digest}"',
                **{name: "true" if value else "false" for name, value in semantics.items()}}
    if artifact.kind == "svd":
        expected["result_order"] = '"u,s,vh"'
    for key, value in expected.items():
        if re.search(r"[{,]\s*" + re.escape(key) + r" = " + re.escape(value) + r"[,}]", attrs) is None:
            raise ValueError(f"Apple GPU native kernel Target IR disagrees on {key}")
    if any(re.search(r"[{,]\s*" + flag + r" = ", attrs) for flag in _FLAG_NAMES - set(semantics)):
        raise ValueError("Apple GPU native kernel Target IR carries a flag its contract does not")
    if any(not isinstance(value, int) or isinstance(value, bool) or not 0 < value <= _INT32_MAX
           for value in scalars.values()):
        raise ValueError("Apple GPU native kernel ABI scalars must be positive int32")
    return AppleGPUKernelProjection(
        kind=artifact.kind, symbol=artifact.symbol, abi=abi, bindings=tuple(bindings),
        inputs=len(artifact.input_names), shapes=shapes, scalars=dict(scalars),
        semantics=dict(semantics), schedule_digest=artifact.schedule_digest,
    )


__all__ = [
    "APPLE_GPU_KERNEL_ENTRIES", "APPLE_GPU_KERNEL_OPS", "AppleGPUKernelProjection",
    "AppleGPUKernelRequest", "ScheduledAppleGPUKernelArtifact", "admit",
    "apple_gpu_value_lowering_available", "lower_scheduled_apple_gpu_kernel", "lowering_available",
    "supports_scheduled_apple_gpu_kernel", "verify_apple_gpu_kernel_projection",
]
