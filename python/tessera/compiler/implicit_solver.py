"""Executable content-addressed implicit-function differentiation packages.

The shared ``tessera_solver`` dialect owns the general IFT semantics.  This
module binds one deliberately narrow physical family to that chain so x86 and
ROCm can establish numerical and performance evidence without treating an
arbitrary Python residual callback as compiler IR.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from typing import Any, Sequence

from .scheduled_matmul import find_tessera_opt, run_tessera_opt


_SCHEMA = "tessera.solver_ift.v1"
_RESIDUAL_MODEL = "diagonal_sqrt_v1"
_LINEAR_SOLVER = "diagonal_matrix_free_v1"
_HASH_RE = re.compile(r'tessera\.schedule_hash = "([0-9a-f]{64})"')


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _residual_source(tensor_type: str) -> str:
    return f"""func.func private @sqrt_residual(%theta: {tensor_type}, %x: {tensor_type}) -> {tensor_type} {{
    %xx = arith.mulf %x, %x : {tensor_type}
    %r = arith.subf %xx, %theta : {tensor_type}
    return %r : {tensor_type}
  }}"""


def build_solver_ift_contract(
    *, target: str, shape: Sequence[int], product_mode: str = "vjp"
) -> dict[str, Any]:
    if target not in {"x86", "rocm_gfx1151"}:
        raise ValueError(
            "solver IFT physical packages support x86 and rocm_gfx1151; unmeasured architectures fail closed"
        )
    normalized = tuple(int(dim) for dim in shape)
    if not normalized or any(dim <= 0 for dim in normalized):
        raise ValueError("solver IFT requires a positive static shape")
    if product_mode not in {"jvp", "vjp"}:
        raise ValueError("solver IFT product_mode must be jvp or vjp")
    architecture = "avx512" if target == "x86" else "gfx1151"
    tensor_type = "tensor<" + "x".join(map(str, normalized)) + "xf32>"
    residual_identity = {
        "model": _RESIDUAL_MODEL,
        "expression": "x*x-theta",
        "parameter_order": ["theta"],
        "solution_order": ["x"],
        "dtype": "f32",
        "shape": list(normalized),
        "source": _residual_source(tensor_type),
    }
    body: dict[str, Any] = {
        "schema": _SCHEMA,
        "target": target,
        "architecture": architecture,
        "shape": list(normalized),
        "storage": "f32",
        "accumulation": "f32",
        "residual": {**residual_identity, "digest": _digest(residual_identity)},
        "linear_solve": {
            "algorithm": _LINEAR_SOLVER,
            "operator": "dR_dx",
            "transpose": product_mode == "vjp",
            "materializes_matrix": False,
        },
        "residual_adjoint": {"wrt": "parameter", "scale": -1.0},
        "phase_outputs": ["residual", "linear_solution", "parameter_cotangent"],
        "workgroup_size": 1 if target == "x86" else 256,
    }
    if product_mode == "jvp":
        body["product_mode"] = "jvp"
    return {**body, "artifact_hash": _digest(body)}


@dataclass(frozen=True)
class ScheduledSolverIFTArtifact:
    target: str
    architecture: str
    shape: tuple[int, ...]
    contract: dict[str, Any]
    shared_solver_ir: str
    schedule_ir: str
    tile_ir: str
    product_mode: str = "vjp"

    @property
    def artifact_hash(self) -> str:
        return str(self.contract["artifact_hash"])

    def runtime_metadata(self) -> dict[str, Any]:
        """Return the exact physical contract consumed by a native runtime."""
        return {
            "target": "x86" if self.target == "x86" else "rocm",
            "compiler_path": ("x86_solver_ift_compiled" if self.target == "x86" else "rocm_solver_ift_compiled"),
            "executable": True,
            "execution_kind": ("native_cpu" if self.target == "x86" else "native_gpu"),
            "arg_names": ["parameter", "solution", "cotangent"],
            "scheduled_solver_ift": self.contract,
        }

    def validate(self) -> None:
        expected = build_solver_ift_contract(
            target=self.target, shape=self.shape, product_mode=self.product_mode
        )
        if self.contract != expected:
            raise ValueError("solver IFT artifact contract is stale")
        for marker in (
            "tessera_solver.residual",
            "tessera_solver.linear_solve",
            "tessera_solver.residual_adjoint",
        ):
            if marker not in self.shared_solver_ir:
                raise ValueError(f"shared solver IFT chain lost {marker}")
        mode_marker = (
            'tessera_solver.residual_adjoint' if self.product_mode == "vjp"
            else 'tessera_solver.residual_jvp'
        )
        if mode_marker not in self.shared_solver_ir:
            raise ValueError(f"shared solver product lost {mode_marker}")
        if self.schedule_ir.count("schedule.solver_ift") != 1:
            raise ValueError("solver IFT requires one typed Schedule operation")
        if self.tile_ir.count("tile.solver_ift_kernel") != 1:
            raise ValueError("solver IFT requires one launch-level Tile artifact")
        if "schedule." in self.tile_ir:
            raise ValueError("solver IFT Tile artifact retains Schedule IR")
        if _HASH_RE.findall(self.tile_ir) != [self.artifact_hash]:
            raise ValueError("solver IFT Tile artifact has stale lineage")


def lower_scheduled_solver_ift(
    *, target: str, shape: Sequence[int], product_mode: str = "vjp"
) -> ScheduledSolverIFTArtifact:
    contract = build_solver_ift_contract(
        target=target, shape=shape, product_mode=product_mode
    )
    tool = find_tessera_opt()
    if tool is None:
        raise RuntimeError("scheduled solver IFT requires production tessera-opt")
    normalized = tuple(int(dim) for dim in shape)
    tensor_type = "tensor<" + "x".join(map(str, normalized)) + "xf32>"
    residual_source = _residual_source(tensor_type)
    shared_input = f"""module {{
  {residual_source}
  func.func @solve(%theta: {tensor_type}) -> {tensor_type} {{
    %x = "tessera_solver.implicit"(%theta) {{residual = @sqrt_residual}} : ({tensor_type}) -> {tensor_type}
    return %x : {tensor_type}
  }}
}}
"""
    solver_option = (
        "--tessera-newton-autodiff=generate-jvp=true"
        if product_mode == "jvp"
        else "--tessera-newton-autodiff"
    )
    shared_solver_ir = run_tessera_opt(tool, shared_input, solver_option)

    digest = str(contract["artifact_hash"])
    residual_digest = str(contract["residual"]["digest"])
    body = {key: value for key, value in contract.items() if key != "artifact_hash"}
    payload = _canonical_json(body)
    architecture = str(contract["architecture"])
    compiler_target = "x86" if target == "x86" else "rocm"
    workgroup = int(contract["workgroup_size"])
    transpose = "true" if product_mode == "vjp" else "false"
    schedule_ir = f'''module attributes {{tessera.target = "{compiler_target}", tessera.arch = "{architecture}"}} {{
  func.func @tessera_solver_ift(%theta: {tensor_type}, %x: {tensor_type}, %cotangent: {tensor_type}) -> ({tensor_type}, {tensor_type}, {tensor_type}) {{
    %phases:3 = schedule.solver_ift %theta, %x, %cotangent {{artifact_hash = "{digest}", lineage_payload = {json.dumps(payload)}, arch = "{architecture}", residual_model = "{_RESIDUAL_MODEL}", residual_digest = "{residual_digest}", linear_solver = "{_LINEAR_SOLVER}", product_mode = "{product_mode}", transpose = {transpose}, wrt = "parameter", adjoint_scale = -1.0 : f32, storage = "f32", accum = "f32", workgroup_size = {workgroup} : i64}} : {tensor_type}, {tensor_type}, {tensor_type} -> {tensor_type}, {tensor_type}, {tensor_type}
    schedule.artifact {{hash = "{digest}", arch = "{architecture}", shape_key = "family=solver_ift;model={_RESIDUAL_MODEL};shape={"x".join(map(str, normalized))};storage=f32", numeric_policy = "f32;matrix_free;transpose;scale=-1"}}
    return %phases#0, %phases#1, %phases#2 : {tensor_type}, {tensor_type}, {tensor_type}
  }}
}}
'''
    tile_ir = run_tessera_opt(tool, schedule_ir, "--tessera-schedule-to-tile")
    artifact = ScheduledSolverIFTArtifact(
        target=target,
        architecture=architecture,
        shape=normalized,
        contract=contract,
        shared_solver_ir=shared_solver_ir,
        schedule_ir=schedule_ir,
        tile_ir=tile_ir,
        product_mode=product_mode,
    )
    artifact.validate()
    return artifact


def diagonal_sqrt_ift_reference(parameter: Any, solution: Any, cotangent: Any) -> tuple[Any, Any, Any]:
    """Numerical oracle for the promoted residual family."""
    residual = solution * solution - parameter
    linear_solution = cotangent / (2.0 * solution)
    return residual, linear_solution, linear_solution.copy()


__all__ = [
    "ScheduledSolverIFTArtifact",
    "build_solver_ift_contract",
    "diagonal_sqrt_ift_reference",
    "lower_scheduled_solver_ift",
]
