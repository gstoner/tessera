"""Native Apple CPU batched GEMM + LAPACK contracts (E2E-REAL-6, 2026-11-04).

One isolated static Apple CPU Graph operation -- ``batched_gemm``,
``tri_solve``, ``cholesky``, ``cholesky_solve``, ``lu``, ``qr`` or ``svd`` --
lowers through the native compiler: ``--tessera-graph-to-schedule`` records a
content-addressed contract
(``src/compiler/programming_model/lib/NativeAppleCPUKernel.h``),
``--tessera-schedule-to-tile`` emits the registered Tile value op, and
``--tile-to-apple_cpu=value=true`` turns it into a verified
``tessera_apple.cpu.call``. The package
(``apple_cpu_native.package_scheduled_apple_cpu_kernel``) is projected from the
*replayed* contract, never from the Python Graph object.

The delegate behind every family is a hand-written Accelerate/LAPACK C ABI
entry: a Decision #28 Tier-3 implementation reached through the declared,
verified Target IR op, which is one lowering path (Decision #31), not a second
authority. The retired Graph-owned constructor is the declared differential
oracle in ``tests/_support/apple_cpu_baseline.py`` (Decision #31(a)).

:func:`supports_scheduled_apple_cpu_kernel` is the host-free admission. It
mirrors what the native contract accepts and normalizes nothing: every semantic
keyword the C ABI does not implement is refused (Decision #21a).
"""

from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass
from typing import Any

from .graph_ir import GraphIRModule

#: Graph op -> (kind, C ABI symbol, delegated routine, inputs, results).
#: Mirrors ``kAppleCPUKernelSpecs`` and the Apple CPU value lowering's symbols.
APPLE_CPU_KERNEL_OPS: dict[str, tuple[str, str, str, int, int]] = {
    "tessera.batched_gemm": ("batched_gemm", "tessera_apple_cpu_gemm_f32_batched",
                             "cblas_sgemm_batched_loop", 2, 1),
    "tessera.tri_solve": ("tri_solve", "tessera_apple_cpu_tri_solve_f32", "lapack_strtrs", 2, 1),
    "tessera.cholesky": ("cholesky", "tessera_apple_cpu_cholesky_f32", "lapack_spotrf", 1, 1),
    "tessera.cholesky_solve": ("cholesky_solve", "tessera_apple_cpu_cholesky_solve_f32",
                               "lapack_spotrs", 2, 1),
    "tessera.lu": ("lu", "tessera_apple_cpu_lu_f32", "lapack_sgetrf", 1, 2),
    "tessera.qr": ("qr", "tessera_apple_cpu_qr_f32", "lapack_sgeqrf", 1, 2),
    "tessera.svd": ("svd", "tessera_apple_cpu_svd_f32", "lapack_sgesvd", 1, 3),
}

#: The semantic flags each family materializes, with their ODS defaults.
_FLAGS: dict[str, dict[str, bool]] = {
    "batched_gemm": {},
    "tri_solve": {"lower": True, "trans": False, "unit_diag": False},
    "cholesky": {"lower": True},
    "cholesky_solve": {"lower": True},
    "lu": {},
    "qr": {"full_matrices": False},
    "svd": {"full_matrices": False},
}

_STORAGE = {"fp32": "f32", "int32": "i32"}
_INT32_MAX = 2**31 - 1
TARGET, ARCHITECTURE = "apple_cpu", "apple_silicon_cpu"


@dataclass(frozen=True)
class AppleCPUKernelRequest:
    """The normalized Graph request one Apple CPU native kernel lowers from."""

    graph_op: str
    kind: str
    symbol: str
    abi: str
    input_names: tuple[str, ...]
    output_names: tuple[str, ...]
    shapes: tuple[tuple[int, ...], ...]
    dtypes: tuple[str, ...]
    semantics: dict[str, bool]


def _static(shape: Any) -> tuple[int, ...]:
    try:
        dims = tuple(int(value) for value in shape)
    except (TypeError, ValueError) as exc:
        raise ValueError("Apple CPU native kernel requires static shapes") from exc
    if not dims or any(value <= 0 or value > _INT32_MAX for value in dims):
        raise ValueError("Apple CPU native kernel requires positive int32 extents")
    return dims


def admit(module: GraphIRModule) -> AppleCPUKernelRequest:
    """Admit one Apple CPU native kernel request or raise ``ValueError``."""
    if len(module.functions) != 1 or len(module.functions[0].body) != 1:
        raise ValueError("Apple CPU native kernel requires one Graph function and operation")
    function = module.functions[0]
    op = function.body[0]
    spec = APPLE_CPU_KERNEL_OPS.get(op.op_name)
    if spec is None:
        raise ValueError(f"{op.op_name} has no Apple CPU native kernel contract")
    kind, symbol, abi, n_inputs, n_results = spec
    args = {arg.name: arg for arg in function.args}
    inputs = tuple(value.removeprefix("%") for value in op.operands)
    outputs = tuple(value.removeprefix("%") for value in function.return_values)
    if (len(inputs) != n_inputs or len(set(inputs)) != n_inputs or set(inputs) != set(args)
            or len(function.args) != n_inputs):
        raise ValueError("Apple CPU native kernel operands must be its distinct entry arguments")
    if (len(outputs) != n_results or len(function.result_types) != n_results
            or len(set(inputs + outputs)) != n_inputs + n_results or "" in outputs):
        raise ValueError("Apple CPU native kernel binds each result to a distinct name")
    shapes = [_static(args[name].ir_type.shape) for name in inputs]
    dtypes = [str(args[name].ir_type.dtype) for name in inputs]
    shapes += [_static(result.shape) for result in function.result_types]
    dtypes += [str(result.dtype) for result in function.result_types]
    if any(dtype not in _STORAGE for dtype in dtypes):
        raise ValueError("Apple CPU native kernel storage is f32 (and i32 LU pivots) only")
    # None-valued keywords mean "not given" (the frontend's spelling).
    raw = {key: value for key, value in op.kwargs.items() if value is not None}
    unknown = sorted(set(raw) - set(_FLAGS[kind]))
    if unknown:
        raise ValueError(f"Apple CPU native {kind} has unsupported keyword(s) {unknown}")
    semantics: dict[str, bool] = {}
    for name, default in _FLAGS[kind].items():
        value = raw.get(name, default)
        if not isinstance(value, bool):
            raise ValueError(f"Apple CPU native {kind} {name} must be a boolean")
        semantics[name] = value

    floats = all(dtype == "fp32" for dtype in dtypes)
    if kind == "batched_gemm":
        lhs, rhs, out = shapes
        if not floats or len(lhs) != 3 or len(rhs) != 3:
            raise ValueError("Apple CPU batched_gemm requires rank-3 f32 operands")
        batch, m, k = lhs
        n = rhs[2]
        if rhs != (batch, k, n) or out != (batch, m, n):
            raise ValueError("Apple CPU batched_gemm requires [B,M,K] x [B,K,N] -> [B,M,N]")
        if max(m * k, k * n, m * n) > _INT32_MAX:
            raise ValueError("Apple CPU batched_gemm per-batch stride exceeds the int32 ABI")
    elif kind in {"tri_solve", "cholesky", "cholesky_solve"}:
        matrix = shapes[0]
        if not floats or len(matrix) != 2 or matrix[0] != matrix[1]:
            raise ValueError(f"Apple CPU {kind} requires a square rank-2 f32 matrix")
        if kind == "cholesky":
            if not semantics["lower"] or shapes[1] != matrix:
                raise ValueError("Apple CPU cholesky implements lower = True into the matrix shape only")
        elif len(shapes[1]) != 2 or shapes[1][0] != matrix[0] or shapes[2] != shapes[1]:
            raise ValueError(f"Apple CPU {kind} requires rhs [N,K] and a result of the rhs shape")
    elif kind == "lu":
        matrix = shapes[0]
        if (len(matrix) != 2 or matrix[0] != matrix[1] or dtypes != ["fp32", "fp32", "int32"]
                or shapes[1] != matrix or shapes[2] != (matrix[0],)):
            raise ValueError("Apple CPU lu requires a square f32 [N,N] -> ([N,N] f32, [N] int32)")
    else:
        matrix = shapes[0]
        if not floats or len(matrix) != 2:
            raise ValueError(f"Apple CPU {kind} requires an f32 rank-2 matrix and f32 results")
        if semantics["full_matrices"]:
            raise ValueError(f"Apple CPU {kind} implements full_matrices = False only")
        m, n = matrix
        r = min(m, n)
        expected = ([(m, n), (n, n)] if kind == "qr" else [(m, r), (r,), (r, n)])
        if (kind == "qr" and m < n) or shapes[1:] != expected:
            raise ValueError(f"Apple CPU {kind} requires the reduced factorization shapes")
    return AppleCPUKernelRequest(
        graph_op=op.op_name, kind=kind, symbol=symbol, abi=abi, input_names=inputs,
        output_names=outputs, shapes=tuple(shapes), dtypes=tuple(dtypes), semantics=semantics,
    )


def supports_scheduled_apple_cpu_kernel(module: GraphIRModule, *, target: str = TARGET) -> bool:
    """Whether the native Apple CPU kernel contract admits ``module``."""
    if target != TARGET:
        return False
    try:
        admit(module)
    except ValueError:
        return False
    return True


@dataclass(frozen=True)
class ScheduledAppleCPUKernelArtifact:
    """One replayable Graph -> Schedule -> Tile -> Target Apple CPU kernel."""

    target: str
    architecture: str
    kind: str
    graph_op: str
    symbol: str
    input_names: tuple[str, ...]
    output_names: tuple[str, ...]
    shapes: tuple[tuple[int, ...], ...]
    dtypes: tuple[str, ...]
    semantics: dict[str, bool]
    graph_ir: str
    schedule_ir: str
    tile_ir: str
    target_ir: str
    schedule_digest: str

    def validate(self) -> None:
        if (self.target, self.architecture) != (TARGET, ARCHITECTURE):
            raise ValueError("Apple CPU native kernel artifact names another target")
        spec = APPLE_CPU_KERNEL_OPS.get(self.graph_op)
        if spec is None or (spec[0], spec[1]) != (self.kind, self.symbol):
            raise ValueError("Apple CPU native kernel artifact kind/symbol disagree with its op")
        if (len(self.input_names) != spec[3] or len(self.output_names) != spec[4]
                or len(self.shapes) != spec[3] + spec[4] or len(self.dtypes) != len(self.shapes)):
            raise ValueError("Apple CPU native kernel artifact arity disagrees with its op")
        if re.fullmatch(r"[0-9a-f]{64}", self.schedule_digest) is None:
            raise ValueError("Apple CPU native kernel artifact requires a schedule digest")


def _boundaries(tool: Any, graph_ir: str) -> tuple[str, str, str]:
    from .scheduled_matmul import run_tessera_opt

    schedule_ir = run_tessera_opt(tool, graph_ir, "--tessera-graph-to-schedule")
    tile_ir = run_tessera_opt(tool, schedule_ir, "--tessera-schedule-to-tile")
    target_ir = run_tessera_opt(tool, tile_ir, "--tile-to-apple_cpu=value=true")
    return schedule_ir, tile_ir, target_ir


def lower_scheduled_apple_cpu_kernel(
    module: GraphIRModule, *, target: str = TARGET,
) -> ScheduledAppleCPUKernelArtifact:
    """Graph -> Schedule -> Tile -> Target for one admitted request."""
    from .scheduled_matmul import find_tessera_opt

    if target != TARGET:
        raise ValueError("Apple CPU native kernel lowers for apple_cpu only")
    request = admit(module)
    tool = find_tessera_opt()
    if tool is None:
        raise RuntimeError("Apple CPU native kernel requires production tessera-opt")
    targeted = copy.deepcopy(module)
    targeted.module_attrs["tessera.target"] = f'"{TARGET}"'
    targeted.module_attrs["tessera.arch"] = f'"{ARCHITECTURE}"'
    targeted.module_attrs["tessera.launch_bindings"] = json.dumps(
        list(request.input_names + request.output_names))
    graph_ir = targeted.to_mlir(target=TARGET, canonical=True)
    schedule_ir, tile_ir, target_ir = _boundaries(tool, graph_ir)
    digests = re.findall(r'schedule\.artifact .*?hash = "([0-9a-f]{64})"', schedule_ir)
    if len(digests) != 1:
        raise ValueError("Apple CPU native kernel Schedule requires exactly one artifact record")
    return ScheduledAppleCPUKernelArtifact(
        target=TARGET, architecture=ARCHITECTURE, kind=request.kind, graph_op=request.graph_op,
        symbol=request.symbol, input_names=request.input_names, output_names=request.output_names,
        shapes=request.shapes, dtypes=request.dtypes, semantics=dict(request.semantics),
        graph_ir=graph_ir, schedule_ir=schedule_ir, tile_ir=tile_ir, target_ir=target_ir,
        schedule_digest=digests[0],
    )


@dataclass(frozen=True)
class AppleCPUKernelProjection:
    kind: str
    symbol: str
    abi: str
    bindings: tuple[str, ...]
    inputs: int
    shapes: tuple[tuple[int, ...], ...]
    dtypes: tuple[str, ...]
    scalars: dict[str, int]
    semantics: dict[str, bool]
    schedule_digest: str


def verify_apple_cpu_kernel_projection(artifact: ScheduledAppleCPUKernelArtifact) -> AppleCPUKernelProjection:
    """Replay every boundary and project the serialized contract.

    Each stored stage must be byte-equal to the production compiler's replay of
    the stage before it, and every descriptor field the artifact carries must
    equal the contract the native Schedule serialized. The Target IR must hold
    exactly one verified ``tessera_apple.cpu.call`` naming the contract's
    symbol, routine, semantic flags and schedule digest.
    """
    from .native_x86_kernel import parse_contract
    from .scheduled_matmul import find_tessera_opt

    artifact.validate()
    tool = find_tessera_opt()
    if tool is None:
        raise RuntimeError("Apple CPU native kernel packaging requires native Schedule replay")
    schedule_ir, tile_ir, target_ir = _boundaries(tool, artifact.graph_ir)
    if schedule_ir != artifact.schedule_ir:
        raise ValueError("Apple CPU native kernel Schedule disagrees with Graph replay")
    if tile_ir != artifact.tile_ir:
        raise ValueError("Apple CPU native kernel Tile disagrees with Schedule replay")
    if target_ir != artifact.target_ir:
        raise ValueError("Apple CPU native kernel Target IR disagrees with Tile replay")
    header = re.match(r'\s*module attributes \{([^{}]*)\}', artifact.tile_ir)
    if header is None or any(
        re.search(r'(?:^|,)\s*' + re.escape(key) + r' = "' + value + r'"(?:,|$)', header[1]) is None
        for key, value in (("tessera.target", TARGET), ("tessera.arch", ARCHITECTURE))
    ):
        raise ValueError("Apple CPU native kernel parent target disagrees")
    contract = parse_contract(artifact.tile_ir, "apple_cpu_kernel")
    if re.findall(r'tessera\.schedule_hash = "([0-9a-f]{64})"', artifact.tile_ir) != [artifact.schedule_digest]:
        raise ValueError("Apple CPU native kernel Tile artifact has a stale schedule digest")
    tile_ops = re.findall(r"(?m)^\s*(?:%\S+ = )?(tile\.[A-Za-z_0-9]+)", artifact.tile_ir)
    if tile_ops != [f"tile.{artifact.kind}"] or "tessera." + artifact.kind + " " in artifact.tile_ir:
        raise ValueError("Apple CPU native kernel Tile artifact requires exactly its one value op")
    bindings = contract.get("bindings")
    shapes = tuple(tuple(shape) for shape in contract.get("shapes", ()))
    storage = tuple(contract.get("storage", ()))
    dtypes = tuple({"f32": "fp32", "i32": "int32"}.get(s, "") for s in storage)
    semantics = contract.get("semantics")
    scalars = contract.get("scalars")
    if (not isinstance(bindings, list) or not isinstance(semantics, dict)
            or not isinstance(scalars, dict) or contract.get("layout") != "row_major"
            or contract.get("framework") != "Accelerate"):
        raise ValueError("Apple CPU native kernel contract is malformed")
    if (contract.get("kind") != artifact.kind or contract.get("graph_op") != artifact.graph_op
            or contract.get("symbol") != artifact.symbol
            or tuple(bindings) != artifact.input_names + artifact.output_names
            or contract.get("inputs") != len(artifact.input_names)
            or shapes != artifact.shapes or dtypes != artifact.dtypes
            or semantics != artifact.semantics):
        raise ValueError("Apple CPU native kernel descriptor fields disagree with native IR")
    abi = str(contract.get("abi", ""))
    calls = re.findall(r"(?m)^\s*(?:%\S+ = )?(tessera_apple\.[A-Za-z_.0-9]+)(.*)$", artifact.target_ir)
    if len(calls) != 1 or calls[0][0] != "tessera_apple.cpu.call" or "tile." in artifact.target_ir:
        raise ValueError("Apple CPU native kernel Target IR requires exactly one tessera_apple.cpu.call")
    attrs = calls[0][1]
    expected = {"op_kind": f'"{artifact.kind}"', "symbol": f'"{artifact.symbol}"', "abi": f'"{abi}"',
                "status": '"executable"', "framework": '"Accelerate"',
                "tessera.schedule_hash": f'"{artifact.schedule_digest}"',
                **{name: "true" if value else "false" for name, value in semantics.items()}}
    for key, value in expected.items():
        if re.search(r"[{,]\s*" + re.escape(key) + r" = " + re.escape(value) + r"[,}]", attrs) is None:
            raise ValueError(f"Apple CPU native kernel Target IR disagrees on {key}")
    if any(re.search(r"[{,]\s*" + flag + r" = ", attrs) for flag in
           {"lower", "trans", "unit_diag", "full_matrices"} - set(semantics)):
        raise ValueError("Apple CPU native kernel Target IR carries a flag its contract does not")
    if any(not isinstance(value, int) or isinstance(value, bool) or not 0 < value <= _INT32_MAX
           for value in scalars.values()):
        raise ValueError("Apple CPU native kernel ABI scalars must be positive int32")
    return AppleCPUKernelProjection(
        kind=artifact.kind, symbol=artifact.symbol, abi=abi, bindings=tuple(bindings),
        inputs=len(artifact.input_names), shapes=shapes, dtypes=dtypes, scalars=dict(scalars),
        semantics=dict(semantics), schedule_digest=artifact.schedule_digest,
    )


__all__ = [
    "APPLE_CPU_KERNEL_OPS", "AppleCPUKernelProjection", "AppleCPUKernelRequest",
    "ScheduledAppleCPUKernelArtifact", "admit", "lower_scheduled_apple_cpu_kernel",
    "supports_scheduled_apple_cpu_kernel", "verify_apple_cpu_kernel_projection",
]
