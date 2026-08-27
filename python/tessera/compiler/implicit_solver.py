"""Executable content-addressed implicit-function differentiation packages.

The shared ``tessera_solver`` dialect owns the general IFT semantics.  This
module binds one deliberately narrow physical family to that chain so x86,
ROCm, and NVIDIA can establish numerical evidence without treating an
arbitrary Python residual callback as compiler IR.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
import copy
from typing import Any, Sequence

from .graph_ir import GraphIRModule, IROp

from .scheduled_matmul import find_tessera_opt, run_tessera_opt


_SCHEMA = "tessera.solver_ift.v1"
_GENERAL_PRODUCT_SCHEMA = "tessera.solver_product.v2"
_RESIDUAL_MODEL = "diagonal_sqrt_v1"
_LINEAR_SOLVER = "diagonal_matrix_free_v1"
_HASH_RE = re.compile(r'tessera\.schedule_hash = "([0-9a-f]{64})"')
_GENERAL_CHILD_ROLES = (
    "residual", "solution_jvp", "solution_vjp", "parameter_jvp",
    "parameter_vjp",
)

_RESIDUAL_PROGRAM_SCHEMA = "tessera.solver.residual_program.v1"
_NVIDIA_KRYLOV_SCHEMA = "tessera.nvidia.krylov.v1"
_NVIDIA_DENSE_KRYLOV_SCHEMA = "tessera.nvidia.krylov.dense.v2"
_RESIDUAL_BINARY_OPS = frozenset({
    "tessera.add", "tessera.sub", "tessera.mul", "tessera.div",
})
_RESIDUAL_UNARY_OPS = frozenset({
    "tessera.sqrt", "tessera.reciprocal", "tessera.exp", "tessera.log",
    "tessera.tanh", "tessera.sigmoid", "tessera.sin", "tessera.cos",
})
_RESIDUAL_REDUCTION_OPS = frozenset({"tessera.sum", "tessera.mean"})
_RESIDUAL_MATMUL_OPS = frozenset({"tessera.matmul", "tessera.gemm"})
_RESIDUAL_COMPARE_OPS = frozenset({
    "tessera.equal", "tessera.not_equal", "tessera.less",
    "tessera.less_equal", "tessera.greater", "tessera.greater_equal",
})
_MAX_RESIDUAL_REGION_UNROLL = 256


def _tensor_contract(ir_type: object) -> tuple[tuple[int | None, ...], str]:
    text = str(ir_type)
    match = re.fullmatch(r"tensor<(.+)x(f32|f16|bf16|i1)>", text)
    if match is None:
        raise ValueError(f"automatic residual packaging requires a ranked tensor; got {text}")
    dims_text, dtype = match.groups()
    dims: list[int | None] = []
    for token in dims_text.split("x"):
        if token == "?":
            dims.append(None)
        else:
            try:
                dim = int(token)
            except ValueError as exc:
                raise ValueError(f"automatic residual packaging rejected tensor dimension {token!r}") from exc
            if dim <= 0:
                raise ValueError("automatic residual packaging requires positive dimensions")
            dims.append(dim)
    return tuple(dims), dtype


def _shape_fits_contract(
    declared: tuple[int | None, ...], bounds: tuple[int, ...]
) -> bool:
    return len(declared) == len(bounds) and all(
        dim is None or dim == bound for dim, bound in zip(declared, bounds)
    )


def _flatten_bounded_regions(
    operations: Sequence[IROp], output: str
) -> tuple[list[IROp], str]:
    """Inline pure bounded regions with capture-safe SSA renaming.

    This is a compiler transform, not host execution: the exact expanded body is
    subsequently differentiated and hashed into every child package. Pure
    data-dependent ``if`` and bounded ``while`` regions become explicit
    compare/selection SSA, so every child recomputes the same digest-bound
    predicate rather than relying on a host callback or untracked tape.
    """
    flattened: list[IROp] = []
    aliases: dict[str, str] = {}
    serial = 0

    def resolve(name: str, local: dict[str, str]) -> str:
        bare = name.removeprefix("%")
        while bare in local and local[bare] != bare:
            bare = local[bare]
        return bare

    def visit(items: Sequence[IROp], local: dict[str, str]) -> None:
        nonlocal serial
        for op in items:
            result_names = op.result_names
            if len(result_names) != 1:
                raise ValueError("automatic residual packaging requires single-result operations")
            original_result = result_names[0]
            if op.op_name == "tessera.control_for":
                trip = int(op.kwargs.get("_trip", -1))
                if trip < 0:
                    raise ValueError("automatic residual packaging requires a non-negative bounded for region")
                if trip > _MAX_RESIDUAL_REGION_UNROLL:
                    raise ValueError(
                        "automatic residual packaging rejects control_for bounds "
                        f"above {_MAX_RESIDUAL_REGION_UNROLL}"
                    )
                body = op.kwargs.get("_body")
                carry_ssa = str(op.kwargs.get("_carry_ssa", ""))
                next_ssa = str(op.kwargs.get("_next_ssa", ""))
                if not isinstance(body, list) or not carry_ssa or not next_ssa or len(op.operands) != 1:
                    raise ValueError("automatic residual packaging rejected malformed control_for")
                carry = resolve(op.operands[0], local)
                for _ in range(trip):
                    iteration = dict(local)
                    iteration[carry_ssa] = carry
                    visit(body, iteration)
                    carry = resolve(next_ssa, iteration)
                local[original_result] = carry
                continue
            if op.op_name == "tessera.control_if":
                if len(op.result_names) != 1:
                    raise ValueError(
                        "automatic residual packaging requires a native variadic "
                        "region product for multi-state control_if"
                    )
                then_body = op.kwargs.get("_then_body")
                else_body = op.kwargs.get("_else_body")
                then_ssa = str(op.kwargs.get("_then_ssa", ""))
                else_ssa = str(op.kwargs.get("_else_ssa", ""))
                flag_ssa = str(op.kwargs.get("_flag_ssa", ""))
                if (
                    not isinstance(then_body, list)
                    or not isinstance(else_body, list)
                    or not then_ssa or not else_ssa or not flag_ssa
                ):
                    raise ValueError("automatic residual packaging rejected malformed control_if")
                if not op.operand_types or op.operand_types[0] != "tensor<1xi1>":
                    raise ValueError(
                        "automatic residual packaging requires a scalar tensor<1xi1> control_if predicate"
                    )
                then_env = dict(local)
                else_env = dict(local)
                visit(then_body, then_env)
                visit(else_body, else_env)
                selected = f"region{serial}_{original_result}"
                serial += 1
                flattened.append(IROp(
                    result=selected,
                    op_name="tessera.where",
                    operands=[
                        f"%{resolve(flag_ssa, local)}",
                        f"%{resolve(then_ssa, then_env)}",
                        f"%{resolve(else_ssa, else_env)}",
                    ],
                    operand_types=[],
                    result_type=op.result_type,
                    kwargs={"predicate_replay": "pure_recompute"},
                ))
                local[original_result] = selected
                continue
            if op.op_name == "tessera.control_while":
                max_iters = int(op.kwargs.get("_max_iters", -1))
                body = op.kwargs.get("_body")
                cond = op.kwargs.get("_cond")
                carry_ssa = str(op.kwargs.get("_carry_ssa", ""))
                next_ssa = str(op.kwargs.get("_next_ssa", ""))
                pred_ssa = str(op.kwargs.get("_pred_ssa", ""))
                if max_iters < 0 or max_iters > _MAX_RESIDUAL_REGION_UNROLL:
                    raise ValueError(
                        "automatic residual packaging requires control_while bounds "
                        f"in [0, {_MAX_RESIDUAL_REGION_UNROLL}]"
                    )
                if (
                    not isinstance(body, list) or not isinstance(cond, list)
                    or not carry_ssa or not next_ssa or not pred_ssa
                    or len(op.operands) != 1
                ):
                    raise ValueError("automatic residual packaging rejected malformed control_while")
                predicate_defs = [
                    item for item in cond if pred_ssa in item.result_names
                ]
                if (
                    len(predicate_defs) != 1
                    or predicate_defs[0].result_type != "tensor<1xi1>"
                ):
                    raise ValueError(
                        "automatic residual packaging requires a scalar tensor<1xi1> control_while predicate"
                    )
                carry = resolve(op.operands[0], local)
                for _ in range(max_iters):
                    iteration = dict(local)
                    iteration[carry_ssa] = carry
                    visit(cond, iteration)
                    predicate = resolve(pred_ssa, iteration)
                    visit(body, iteration)
                    candidate = resolve(next_ssa, iteration)
                    selected = f"region{serial}_{original_result}"
                    serial += 1
                    flattened.append(IROp(
                        result=selected,
                        op_name="tessera.where",
                        operands=[f"%{predicate}", f"%{candidate}", f"%{carry}"],
                        operand_types=[],
                        result_type=op.result_type,
                        kwargs={"predicate_replay": "pure_recompute"},
                    ))
                    carry = selected
                local[original_result] = carry
                continue
            if op.op_name.startswith("tessera.control_") or op.kwargs.get("_region") is not None:
                raise ValueError(
                    "automatic residual packaging rejected an unsupported control region"
                )
            new_result = f"region{serial}_{original_result}"
            serial += 1
            cloned = copy.deepcopy(op)
            cloned.result = new_result
            cloned.operands = [f"%{resolve(name, local)}" for name in op.operands]
            flattened.append(cloned)
            local[original_result] = new_result

    visit(operations, aliases)
    return flattened, resolve(output, aliases)


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


class _ResidualProgramBuilder:
    """Small SSA builder for target-native residual/product programs."""

    def __init__(self) -> None:
        self.steps: list[dict[str, Any]] = []
        self._next_id = 0

    def emit(self, op_name: str, *inputs: str, **kwargs: Any) -> str:
        result = f"p{self._next_id}"
        self._next_id += 1
        step: dict[str, Any] = {
            "id": result, "kind": "op", "op_name": op_name,
            "inputs": list(inputs),
        }
        if kwargs:
            step["kwargs"] = kwargs
        self.steps.append(step)
        return result

    def constant(self, value: float) -> str:
        result = f"p{self._next_id}"
        self._next_id += 1
        self.steps.append({
            "id": result, "kind": "constant", "value": float(value),
        })
        return result

    def zero_like(self, value: str) -> str:
        return self.emit("tessera.zeros_like", value)

    def add(self, lhs: str, rhs: str) -> str:
        return self.emit("tessera.add", lhs, rhs)

    def mul(self, lhs: str, rhs: str) -> str:
        return self.emit("tessera.mul", lhs, rhs)

    def neg(self, value: str) -> str:
        return self.mul(value, self.constant(-1.0))


def _residual_op(builder: _ResidualProgramBuilder, op: IROp,
                 operands: list[str]) -> str:
    if op.attrs:
        raise ValueError(
            f"automatic residual packaging rejects attributes on {op.op_name}"
        )
    if op.op_name in _RESIDUAL_BINARY_OPS and len(operands) == 2:
        return builder.emit(op.op_name, *operands)
    if op.op_name in _RESIDUAL_UNARY_OPS and len(operands) == 1:
        return builder.emit(op.op_name, *operands)
    if op.op_name in _RESIDUAL_MATMUL_OPS and len(operands) == 2:
        return builder.emit("tessera.matmul", *operands, **_matmul_product_kwargs(op))
    if op.op_name in _RESIDUAL_COMPARE_OPS and len(operands) == 2:
        if op.kwargs:
            raise ValueError("automatic residual packaging rejects comparison attributes")
        return builder.emit(
            op.op_name, *operands, derivative_boundary="reject_equal"
        )
    if op.op_name == "tessera.where" and len(operands) == 3:
        if set(op.kwargs) - {"predicate_replay"}:
            raise ValueError("automatic residual packaging rejects selection attributes")
        if op.kwargs.get("predicate_replay") != "pure_recompute":
            raise ValueError("automatic residual packaging requires explicit predicate replay")
        return builder.emit(
            op.op_name, *operands, predicate_replay="pure_recompute"
        )
    if op.op_name in _RESIDUAL_REDUCTION_OPS and len(operands) == 1:
        allowed = {"axis", "keepdims"}
        if set(op.kwargs) - allowed:
            raise ValueError(
                f"automatic residual packaging rejects reduction attributes {set(op.kwargs) - allowed}"
            )
        axis = op.kwargs.get("axis")
        if isinstance(axis, tuple):
            axis = list(axis)
        return builder.emit(
            op.op_name, operands[0], axis=axis,
            keepdims=bool(op.kwargs.get("keepdims", False)),
        )
    if op.op_name == "tessera.transpose" and len(operands) == 1:
        allowed = {"axes", "perm"}
        if set(op.kwargs) - allowed:
            raise ValueError("automatic residual packaging rejects transpose attributes")
        axes = op.kwargs.get("axes", op.kwargs.get("perm"))
        return builder.emit("tessera.transpose", operands[0], axes=axes)
    raise ValueError(
        "automatic residual packaging supports typed pointwise, reduction, "
        "transpose, comparison/selection, and rank-2 matmul operations; "
        f"got {op.op_name!r} with {len(operands)} operands"
    )


def _matmul_storage(op: IROp) -> str:
    for operand_type in op.operand_types:
        match = re.search(r"x(f32|f16|bf16)>", operand_type)
        if match is not None:
            return match.group(1)
    return "f32"


def _matmul_product_kwargs(op: IROp) -> dict[str, Any]:
    allowed = {"storage", "accumulation", "numeric_policy", "math_mode"}
    unknown = set(op.kwargs) - allowed
    if unknown:
        raise ValueError(
            f"automatic residual packaging rejects matmul attributes {unknown}"
        )
    accumulation = str(op.kwargs.get("accumulation", "f32"))
    if accumulation not in {"f32", "fp32"}:
        raise ValueError("automatic residual matmul requires f32 accumulation")
    result: dict[str, Any] = {
        "storage": _matmul_storage(op), "accumulation": "f32",
    }
    policy = op.kwargs.get("numeric_policy")
    if policy is not None:
        if not isinstance(policy, dict):
            raise ValueError("automatic residual matmul numeric_policy must be a dictionary")
        result["numeric_policy"] = copy.deepcopy(policy)
    elif op.kwargs.get("math_mode") is not None:
        result["numeric_policy"] = {"math_mode": str(op.kwargs["math_mode"])}
    return result


def _product_kwargs(op: IROp) -> dict[str, Any]:
    if op.op_name in _RESIDUAL_MATMUL_OPS:
        return _matmul_product_kwargs(op)
    return dict(op.kwargs)


def _forward_product(builder: _ResidualProgramBuilder, op_name: str,
                     primal_inputs: list[str], tangent_inputs: list[str],
                     primal_output: str, kwargs: dict[str, Any]) -> str:
    if op_name == "tessera.add":
        return builder.add(tangent_inputs[0], tangent_inputs[1])
    if op_name == "tessera.sub":
        return builder.emit("tessera.sub", tangent_inputs[0], tangent_inputs[1])
    if op_name == "tessera.mul":
        return builder.add(
            builder.mul(tangent_inputs[0], primal_inputs[1]),
            builder.mul(primal_inputs[0], tangent_inputs[1]),
        )
    if op_name == "tessera.div":
        numerator = builder.emit(
            "tessera.sub",
            builder.mul(tangent_inputs[0], primal_inputs[1]),
            builder.mul(primal_inputs[0], tangent_inputs[1]),
        )
        return builder.emit(
            "tessera.div", numerator,
            builder.mul(primal_inputs[1], primal_inputs[1]),
        )
    if op_name in _RESIDUAL_MATMUL_OPS:
        return builder.add(
            builder.emit("tessera.matmul", tangent_inputs[0], primal_inputs[1], **kwargs),
            builder.emit("tessera.matmul", primal_inputs[0], tangent_inputs[1], **kwargs),
        )
    if op_name in _RESIDUAL_COMPARE_OPS:
        return builder.zero_like(primal_output)
    if op_name == "tessera.where":
        return builder.emit(
            op_name, primal_inputs[0], tangent_inputs[1], tangent_inputs[2],
            predicate_replay="pure_recompute",
        )
    if op_name in _RESIDUAL_REDUCTION_OPS:
        return builder.emit(op_name, tangent_inputs[0], **kwargs)
    if op_name == "tessera.transpose":
        return builder.emit(op_name, tangent_inputs[0], **kwargs)
    dx = tangent_inputs[0]
    x = primal_inputs[0]
    if op_name == "tessera.sqrt":
        return builder.emit(
            "tessera.div", dx,
            builder.mul(builder.constant(2.0), primal_output),
        )
    if op_name == "tessera.reciprocal":
        return builder.neg(builder.emit(
            "tessera.div", dx, builder.mul(x, x)
        ))
    if op_name == "tessera.exp":
        return builder.mul(primal_output, dx)
    if op_name == "tessera.log":
        return builder.emit("tessera.div", dx, x)
    if op_name == "tessera.tanh":
        scale = builder.emit(
            "tessera.sub", builder.constant(1.0),
            builder.mul(primal_output, primal_output),
        )
        return builder.mul(scale, dx)
    if op_name == "tessera.sigmoid":
        scale = builder.mul(
            primal_output,
            builder.emit("tessera.sub", builder.constant(1.0), primal_output),
        )
        return builder.mul(scale, dx)
    if op_name == "tessera.sin":
        return builder.mul(builder.emit("tessera.cos", x), dx)
    if op_name == "tessera.cos":
        return builder.neg(builder.mul(builder.emit("tessera.sin", x), dx))
    raise ValueError(f"no forward residual product for {op_name}")


def _reverse_contributions(builder: _ResidualProgramBuilder, op_name: str,
                           primal_inputs: list[str], primal_output: str,
                           output_adjoint: str,
                           kwargs: dict[str, Any]) -> list[str | None]:
    def unbroadcast(value: str, witness: str) -> str:
        return builder.emit("tessera.unbroadcast_like", value, witness)

    if op_name == "tessera.add":
        return [unbroadcast(output_adjoint, primal_inputs[0]),
                unbroadcast(output_adjoint, primal_inputs[1])]
    if op_name == "tessera.sub":
        return [unbroadcast(output_adjoint, primal_inputs[0]),
                unbroadcast(builder.neg(output_adjoint), primal_inputs[1])]
    if op_name == "tessera.mul":
        return [unbroadcast(builder.mul(output_adjoint, primal_inputs[1]), primal_inputs[0]),
                unbroadcast(builder.mul(output_adjoint, primal_inputs[0]), primal_inputs[1])]
    if op_name == "tessera.div":
        lhs = builder.emit("tessera.div", output_adjoint, primal_inputs[1])
        rhs = builder.neg(builder.emit(
            "tessera.div",
            builder.mul(output_adjoint, primal_inputs[0]),
            builder.mul(primal_inputs[1], primal_inputs[1]),
        ))
        return [unbroadcast(lhs, primal_inputs[0]),
                unbroadcast(rhs, primal_inputs[1])]
    if op_name in _RESIDUAL_MATMUL_OPS:
        lhs_t = builder.emit("tessera.transpose", primal_inputs[0], axes=[1, 0])
        rhs_t = builder.emit("tessera.transpose", primal_inputs[1], axes=[1, 0])
        return [
            builder.emit("tessera.matmul", output_adjoint, rhs_t, **kwargs),
            builder.emit("tessera.matmul", lhs_t, output_adjoint, **kwargs),
        ]
    if op_name in _RESIDUAL_COMPARE_OPS:
        return [None for _ in primal_inputs]
    if op_name == "tessera.where":
        zero = builder.zero_like(output_adjoint)
        then_adjoint = builder.emit(
            op_name, primal_inputs[0], output_adjoint, zero,
            predicate_replay="pure_recompute",
        )
        else_adjoint = builder.emit(
            op_name, primal_inputs[0], zero, output_adjoint,
            predicate_replay="pure_recompute",
        )
        return [
            None,
            unbroadcast(then_adjoint, primal_inputs[1]),
            unbroadcast(else_adjoint, primal_inputs[2]),
        ]
    if op_name in _RESIDUAL_REDUCTION_OPS:
        expanded = builder.emit(
            "tessera.broadcast_like", output_adjoint, primal_inputs[0]
        )
        if op_name == "tessera.mean":
            expanded = builder.emit(
                "tessera.mean_adjoint", expanded, primal_inputs[0], **kwargs
            )
        return [expanded]
    if op_name == "tessera.transpose":
        axes = kwargs.get("axes")
        if axes is None:
            inverse = None
        else:
            inverse = [0] * len(axes)
            for index, axis in enumerate(axes):
                inverse[int(axis)] = index
        return [builder.emit("tessera.transpose", output_adjoint, axes=inverse)]
    x = primal_inputs[0]
    if op_name == "tessera.sqrt":
        return [builder.emit(
            "tessera.div", output_adjoint,
            builder.mul(builder.constant(2.0), primal_output),
        )]
    if op_name == "tessera.reciprocal":
        return [builder.neg(builder.emit(
            "tessera.div", output_adjoint, builder.mul(x, x)
        ))]
    if op_name == "tessera.exp":
        return [builder.mul(output_adjoint, primal_output)]
    if op_name == "tessera.log":
        return [builder.emit("tessera.div", output_adjoint, x)]
    if op_name == "tessera.tanh":
        return [builder.mul(output_adjoint, builder.emit(
            "tessera.sub", builder.constant(1.0),
            builder.mul(primal_output, primal_output),
        ))]
    if op_name == "tessera.sigmoid":
        return [builder.mul(output_adjoint, builder.mul(
            primal_output,
            builder.emit("tessera.sub", builder.constant(1.0), primal_output),
        ))]
    if op_name == "tessera.sin":
        return [builder.mul(output_adjoint, builder.emit("tessera.cos", x))]
    if op_name == "tessera.cos":
        return [builder.neg(builder.mul(
            output_adjoint, builder.emit("tessera.sin", x)
        ))]
    raise ValueError(f"no reverse residual product for {op_name}")


def _compile_residual_program(
    *, operations: Sequence[IROp], output: str, parameter_name: str,
    solution_name: str, role: str,
) -> tuple[dict[str, Any], list[str]]:
    is_product = role != "residual"
    builder = _ResidualProgramBuilder()
    env = {parameter_name: "parameter", solution_name: "solution"}
    tangents = {
        parameter_name: "vector" if role == "parameter_jvp" else builder.zero_like("parameter"),
        solution_name: "vector" if role == "solution_jvp" else builder.zero_like("solution"),
    }
    records: list[tuple[IROp, list[str], str]] = []
    for op in operations:
        if len(op.result_names) != 1:
            raise ValueError("automatic residual packaging requires single-result operations")
        operand_names = [name.removeprefix("%") for name in op.operands]
        if any(name not in env for name in operand_names):
            raise ValueError(f"residual operation {op.op_name} references an unknown SSA value")
        primal_inputs = [env[name] for name in operand_names]
        primal_output = _residual_op(builder, op, primal_inputs)
        result_name = op.result_names[0]
        env[result_name] = primal_output
        records.append((op, operand_names, primal_output))
        if role.endswith("_jvp"):
            tangent_inputs = [tangents[name] for name in operand_names]
            tangents[result_name] = _forward_product(
                builder, op.op_name, primal_inputs, tangent_inputs, primal_output,
                _product_kwargs(op),
            )
    if output not in env:
        raise ValueError("residual Graph return does not reference a defined value")
    if role == "residual":
        program_output = env[output]
    elif role.endswith("_jvp"):
        program_output = tangents[output]
    else:
        adjoints: dict[str, str] = {output: "vector"}
        for op, operand_names, primal_output in reversed(records):
            result_name = op.result_names[0]
            output_adjoint = adjoints.get(result_name)
            if output_adjoint is None:
                continue
            contributions = _reverse_contributions(
                builder, op.op_name, [env[name] for name in operand_names],
                primal_output, output_adjoint, _product_kwargs(op),
            )
            for name, contribution in zip(operand_names, contributions):
                if contribution is None:
                    continue
                adjoints[name] = (
                    contribution if name not in adjoints
                    else builder.add(adjoints[name], contribution)
                )
        wanted = parameter_name if role == "parameter_vjp" else solution_name
        program_output = adjoints.get(
            wanted,
            builder.zero_like("parameter" if wanted == parameter_name else "solution"),
        )
    arg_names = ["parameter", "solution"] + (["vector"] if is_product else [])
    body = {
        "schema": _RESIDUAL_PROGRAM_SCHEMA,
        "role": role,
        "arg_names": arg_names,
        "steps": builder.steps,
        "output": program_output,
        "transpose": role.endswith("_vjp"),
        "predicate_replay": {
            "policy": "pure_recompute",
            "derivative_boundary": "reject_equal",
            "selection_count": sum(
                step.get("op_name") == "tessera.where"
                for step in builder.steps
            ),
        },
    }
    return {**body, "program_digest": _digest(body)}, arg_names


def _residual_source(tensor_type: str) -> str:
    return f"""func.func private @sqrt_residual(%theta: {tensor_type}, %x: {tensor_type}) -> {tensor_type} {{
    %xx = arith.mulf %x, %x : {tensor_type}
    %r = arith.subf %xx, %theta : {tensor_type}
    return %r : {tensor_type}
  }}"""


def build_solver_ift_contract(
    *, target: str, shape: Sequence[int], product_mode: str = "vjp"
) -> dict[str, Any]:
    if target not in {"x86", "rocm_gfx1151", "nvidia_sm120"}:
        raise ValueError(
            "solver IFT physical packages support x86, rocm_gfx1151, and "
            "nvidia_sm120; unmeasured architectures fail closed"
        )
    normalized = tuple(int(dim) for dim in shape)
    if not normalized or any(dim <= 0 for dim in normalized):
        raise ValueError("solver IFT requires a positive static shape")
    if product_mode not in {"jvp", "vjp"}:
        raise ValueError("solver IFT product_mode must be jvp or vjp")
    architecture = {
        "x86": "avx512",
        "rocm_gfx1151": "gfx1151",
        "nvidia_sm120": "sm120",
    }[target]
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


def build_general_solver_product_contract(
    *,
    target: str,
    shape: Sequence[int],
    residual_graph_ir: str,
    residual_jvp_ir: str,
    residual_vjp_ir: str,
    product_mode: str,
    linear_solver: str = "gmres",
) -> dict[str, Any]:
    """Bind arbitrary residual products without claiming a native package.

    The diagonal-sqrt package below remains the only architecture-proven
    physical consumer.  This v2 contract closes the semantic/lineage gap for
    general matrix-free solves: a target producer must consume these exact
    Graph product digests and may promote ``artifact_only`` only after its own
    executable and exact-device evidence land.
    """
    if target not in {"x86", "rocm_gfx1151", "nvidia_sm120"}:
        raise ValueError(
            "general solver products support x86, rocm_gfx1151, and "
            "nvidia_sm120 contracts"
        )
    normalized = tuple(int(dim) for dim in shape)
    if not normalized or any(dim <= 0 for dim in normalized):
        raise ValueError("general solver products require a positive static shape")
    if product_mode not in {"jvp", "vjp"}:
        raise ValueError("general solver product_mode must be jvp or vjp")
    if linear_solver not in {"gmres", "cg"}:
        raise ValueError("general solver products require gmres or cg")
    texts = {
        "residual_graph_ir": residual_graph_ir,
        "residual_jvp_ir": residual_jvp_ir,
        "residual_vjp_ir": residual_vjp_ir,
    }
    if any(not text.strip() for text in texts.values()):
        raise ValueError("general solver products require residual, JVP, and VJP IR")
    architecture = {
        "x86": "zen5_avx512",
        "rocm_gfx1151": "gfx1151",
        "nvidia_sm120": "sm120",
    }[target]
    body: dict[str, Any] = {
        "schema": _GENERAL_PRODUCT_SCHEMA,
        "target": target,
        "architecture": architecture,
        "shape": list(normalized),
        "product_mode": product_mode,
        "linear_solver": linear_solver,
        "matrix_free": True,
        "transpose": product_mode == "vjp",
        "residual_graph_digest": hashlib.sha256(
            residual_graph_ir.encode("utf-8")
        ).hexdigest(),
        "residual_jvp_digest": hashlib.sha256(
            residual_jvp_ir.encode("utf-8")
        ).hexdigest(),
        "residual_vjp_digest": hashlib.sha256(
            residual_vjp_ir.encode("utf-8")
        ).hexdigest(),
        "execution_state": "artifact_only",
        "promotion_gate": "architecture_owned_native_package_and_exact_device_packet",
    }
    return {**body, "artifact_hash": _digest(body)}


def build_physical_general_solver_contract(
    *, target: str, shape: Sequence[int], residual_graph_ir: str,
    residual_jvp_ir: str, residual_vjp_ir: str, product_mode: str,
    children: dict[str, dict[str, Any]], linear_solver: str = "gmres",
    tolerance: float = 1.0e-6, max_iterations: int = 64,
    restart: int = 16,
) -> dict[str, Any]:
    """Promote a general residual only when all matrix-free children exist.

    Each child contains serializable native runtime metadata plus explicit
    symbolic bindings.  The parent hashes both, so a residual, product, or
    target package cannot be replaced after scheduling.
    """
    base = build_general_solver_product_contract(
        target=target, shape=shape, residual_graph_ir=residual_graph_ir,
        residual_jvp_ir=residual_jvp_ir, residual_vjp_ir=residual_vjp_ir,
        product_mode=product_mode, linear_solver=linear_solver,
    )
    if set(children) != set(_GENERAL_CHILD_ROLES):
        raise ValueError(
            "physical general solver requires residual, solution_jvp, "
            "solution_vjp, parameter_jvp, and parameter_vjp children"
        )
    if not (0.0 < float(tolerance) < 1.0):
        raise ValueError("physical general solver tolerance must be in (0, 1)")
    if max_iterations <= 0 or restart <= 0 or restart > max_iterations:
        raise ValueError("physical general solver requires 0 < restart <= max_iterations")
    child_records: dict[str, Any] = {}
    expected_target = {
        "x86": "x86",
        "rocm_gfx1151": "rocm",
        "nvidia_sm120": "nvidia_sm120",
    }[target]
    for role in _GENERAL_CHILD_ROLES:
        child = dict(children[role])
        metadata = dict(child.get("metadata") or {})
        bindings = list(child.get("bindings") or [])
        if metadata.get("target") != expected_target or not metadata.get("executable"):
            raise ValueError(f"{role} must be an executable {expected_target} child")
        if len(bindings) != len(metadata.get("arg_names") or []):
            raise ValueError(f"{role} bindings must match its child argument order")
        if any(binding not in {
            "parameter", "solution", "vector", "zero", "one", "minus_one"
        } for binding in bindings):
            raise ValueError(f"{role} has an unsupported symbolic binding")
        child_records[role] = {
            "metadata": metadata,
            "bindings": bindings,
            "digest": _digest({"metadata": metadata, "bindings": bindings}),
        }
    body = {key: value for key, value in base.items() if key != "artifact_hash"}
    body.update({
        "schema": "tessera.solver_product.physical.v1",
        "execution_state": "native_composite",
        "children": child_records,
        "convergence": {
            "tolerance": float(tolerance),
            "max_iterations": int(max_iterations),
            "restart": int(restart),
            "true_residual_check": True,
        },
        "promotion_gate": "exact_device_complete_solve_packet",
    })
    return {**body, "artifact_hash": _digest(body)}


@dataclass(frozen=True)
class PhysicalGeneralSolverArtifact:
    target: str
    shape: tuple[int, ...]
    contract: dict[str, Any]

    @property
    def artifact_hash(self) -> str:
        return str(self.contract["artifact_hash"])

    def runtime_metadata(self) -> dict[str, Any]:
        runtime_target = {
            "x86": "x86",
            "rocm_gfx1151": "rocm",
            "nvidia_sm120": "nvidia_sm120",
        }[self.target]
        compiler_prefix = "nvidia" if self.target == "nvidia_sm120" else runtime_target
        return {
            "target": runtime_target,
            "compiler_path": f"{compiler_prefix}_general_solver_compiled",
            "executable": True,
            "execution_kind": "native_cpu" if self.target == "x86" else "native_gpu",
            "arg_names": ["parameter", "solution", "product"],
            "physical_general_solver": self.contract,
        }


def package_physical_general_solver(
    *, target: str, shape: Sequence[int], residual_graph_ir: str,
    residual_jvp_ir: str, residual_vjp_ir: str, product_mode: str,
    children: dict[str, dict[str, Any]], linear_solver: str = "gmres",
    tolerance: float = 1.0e-6, max_iterations: int = 64,
    restart: int = 16,
) -> PhysicalGeneralSolverArtifact:
    contract = build_physical_general_solver_contract(
        target=target, shape=shape, residual_graph_ir=residual_graph_ir,
        residual_jvp_ir=residual_jvp_ir, residual_vjp_ir=residual_vjp_ir,
        product_mode=product_mode, children=children,
        linear_solver=linear_solver, tolerance=tolerance,
        max_iterations=max_iterations, restart=restart,
    )
    return PhysicalGeneralSolverArtifact(
        target=target, shape=tuple(int(dim) for dim in shape), contract=contract
    )


def compile_physical_general_solver_from_graph(
    module: GraphIRModule, *, target: str, shape: Sequence[int],
    parameter_shape: Sequence[int] | None = None,
    parameter_arg: int = 0, solution_arg: int = 1,
    product_mode: str = "vjp", linear_solver: str = "gmres",
    tolerance: float = 1.0e-6, max_iterations: int = 64,
    restart: int = 16,
) -> PhysicalGeneralSolverArtifact:
    """Generate the five physical solver children from one residual Graph.

    The production envelope is one shape-preserving residual with bounded
    dynamic dimensions and f16/bf16/f32 storage (f32 products). Pointwise,
    reduction, transpose, rank-2 matmul, and statically bounded ``control_for``
    bodies are differentiated from SSA dataflow. Data-dependent regions and
    unsupported dtype/layout transitions fail closed.
    """
    if target not in {"x86", "rocm_gfx1151", "nvidia_sm120"}:
        raise ValueError(
            "automatic residual packaging supports x86, rocm_gfx1151, and "
            "nvidia_sm120"
        )
    if len(module.functions) != 1:
        raise ValueError("automatic residual packaging requires exactly one Graph function")
    verification = module.verify()
    if not verification.ok:
        raise ValueError("automatic residual packaging rejected invalid Graph IR: " + verification.format())
    function = module.functions[0]
    if len(function.args) != 2 or sorted((parameter_arg, solution_arg)) != [0, 1]:
        raise ValueError("automatic residual packaging requires parameter and solution as two distinct arguments")
    if len(function.return_values) != 1 or len(function.result_types) != 1:
        raise ValueError("automatic residual packaging requires one residual result")
    normalized = tuple(int(dim) for dim in shape)
    if not normalized or any(dim <= 0 for dim in normalized):
        raise ValueError("automatic residual packaging requires a positive static shape")
    argument_contracts = [_tensor_contract(argument.ir_type) for argument in function.args]
    if any(dtype not in {"f32", "f16", "bf16"} for _dims, dtype in argument_contracts):
        raise ValueError("automatic residual packaging requires floating parameter and solution tensors")
    solution_declared = argument_contracts[solution_arg][0]
    if not _shape_fits_contract(solution_declared, normalized):
        raise ValueError(
            "automatic residual packaging solution shape must match the static "
            "shape or use bounded-dynamic dimensions"
        )
    parameter_declared = argument_contracts[parameter_arg][0]
    if parameter_shape is None:
        if all(dim is not None for dim in parameter_declared):
            parameter_bounds = tuple(
                dim for dim in parameter_declared if dim is not None
            )
        elif len(parameter_declared) == len(normalized):
            parameter_bounds = normalized
        else:
            raise ValueError(
                "a dynamic parameter with a different rank requires parameter_shape bounds"
            )
    else:
        parameter_bounds = tuple(int(dim) for dim in parameter_shape)
    if not parameter_bounds or any(dim <= 0 for dim in parameter_bounds):
        raise ValueError("automatic residual packaging requires positive parameter bounds")
    if not _shape_fits_contract(parameter_declared, parameter_bounds):
        raise ValueError("automatic residual packaging parameter bounds contradict Graph IR")
    result_dims, result_dtype = _tensor_contract(function.result_types[0])
    if not _shape_fits_contract(result_dims, normalized) or result_dtype != "f32":
        raise ValueError(
            "automatic residual packaging requires an f32 residual matching the solution bounds"
        )
    for op in function.body:
        if op.result_type is not None:
            _tensor_contract(op.result_type)

    parameter_name = function.args[parameter_arg].name.removeprefix("%")
    solution_name = function.args[solution_arg].name.removeprefix("%")
    output = function.return_values[0].removeprefix("%")
    operations, output = _flatten_bounded_regions(function.body, output)
    runtime_target = {
        "x86": "x86",
        "rocm_gfx1151": "rocm",
        "nvidia_sm120": "nvidia_sm120",
    }[target]
    compiler_prefix = "nvidia" if target == "nvidia_sm120" else runtime_target
    execution_kind = "native_cpu" if target == "x86" else "native_gpu"
    programs: dict[str, dict[str, Any]] = {}
    children: dict[str, dict[str, Any]] = {}
    for role in _GENERAL_CHILD_ROLES:
        program, arg_names = _compile_residual_program(
            operations=operations, output=output,
            parameter_name=parameter_name, solution_name=solution_name,
            role=role,
        )
        programs[role] = program
        children[role] = {
            "metadata": {
                "target": runtime_target,
                "compiler_path": f"{compiler_prefix}_solver_graph_compiled",
                "executable": True,
                "execution_kind": execution_kind,
                "arg_names": arg_names,
                "solver_residual_program": program,
            },
            "bindings": arg_names,
        }
    graph_ir = module.to_mlir(canonical=True)
    package = package_physical_general_solver(
        target=target, shape=normalized, residual_graph_ir=graph_ir,
        residual_jvp_ir=_canonical_json({
            "solution_jvp": programs["solution_jvp"],
            "parameter_jvp": programs["parameter_jvp"],
        }),
        residual_vjp_ir=_canonical_json({
            "solution_vjp": programs["solution_vjp"],
            "parameter_vjp": programs["parameter_vjp"],
        }),
        product_mode=product_mode, children=children,
        linear_solver=linear_solver, tolerance=tolerance,
        max_iterations=max_iterations, restart=restart,
    )
    contract = dict(package.contract)
    body = {key: value for key, value in contract.items() if key != "artifact_hash"}
    body["value_contract"] = {
        "shape_bounds": list(normalized),
        "parameter_shape_bounds": list(parameter_bounds),
        "dynamic_dimensions": [
            index for index, dim in enumerate(argument_contracts[solution_arg][0])
            if dim is None
        ],
        "storage_dtypes": {
            "parameter": argument_contracts[parameter_arg][1],
            "solution": argument_contracts[solution_arg][1],
            "residual": result_dtype,
            "product": "f32",
        },
        "accumulation_dtype": "f32",
        "conversion": "explicit_package_boundary_widen",
    }
    contract = {**body, "artifact_hash": _digest(body)}
    return PhysicalGeneralSolverArtifact(
        target=package.target, shape=package.shape, contract=contract
    )


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
        runtime_target = {
            "x86": "x86",
            "rocm_gfx1151": "rocm",
            "nvidia_sm120": "nvidia_sm120",
        }[self.target]
        compiler_prefix = "nvidia" if self.target == "nvidia_sm120" else runtime_target
        return {
            "target": runtime_target,
            "compiler_path": f"{compiler_prefix}_solver_ift_compiled",
            "executable": True,
            "execution_kind": "native_cpu" if self.target == "x86" else "native_gpu",
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
    compiler_target = {
        "x86": "x86",
        "rocm_gfx1151": "rocm",
        "nvidia_sm120": "nvidia",
    }[target]
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


def build_nvidia_krylov_contract(
    *, shape: Sequence[int], storage: str = "f32", algorithm: str = "cg",
    tolerance: float = 1.0e-6, max_iterations: int = 128,
) -> dict[str, Any]:
    """Build the bounded single-launch SM120 Krylov contract.

    The first physical envelope is a positive diagonal SPD operator. All
    evolving vectors and scalar reductions remain in device memory for the
    complete solve; only the immutable diagonal/RHS and final state cross the
    host boundary.
    """
    normalized = tuple(int(dim) for dim in shape)
    if not normalized or any(dim <= 0 for dim in normalized):
        raise ValueError("NVIDIA Krylov requires a positive static shape")
    if storage not in {"f32", "f16", "bf16"}:
        raise ValueError("NVIDIA Krylov storage must be f32, f16, or bf16")
    if algorithm != "cg":
        raise ValueError("NVIDIA device-resident Krylov v1 admits dedicated CG only")
    if not (0.0 < float(tolerance) < 1.0):
        raise ValueError("NVIDIA Krylov tolerance must be in (0, 1)")
    if int(max_iterations) <= 0:
        raise ValueError("NVIDIA Krylov max_iterations must be positive")
    body = {
        "schema": _NVIDIA_KRYLOV_SCHEMA,
        "target": "nvidia_sm120",
        "architecture": "sm120",
        "shape": list(normalized),
        "operator": "positive_diagonal_spd_v1",
        "algorithm": algorithm,
        "storage": storage,
        "accumulation": "f32",
        "state": ["solution", "residual", "direction", "matvec"],
        "state_residency": "single_launch_device_resident",
        "true_residual_check": True,
        "tolerance": float(tolerance),
        "max_iterations": int(max_iterations),
        "workgroup_size": 256,
        "performance_eligible": False,
    }
    return {**body, "artifact_hash": _digest(body)}


@dataclass(frozen=True)
class NvidiaKrylovArtifact:
    shape: tuple[int, ...]
    contract: dict[str, Any]

    @property
    def artifact_hash(self) -> str:
        return str(self.contract["artifact_hash"])

    def runtime_metadata(self) -> dict[str, Any]:
        return {
            "target": "nvidia_sm120",
            "compiler_path": "nvidia_krylov_solver_compiled",
            "executable": True,
            "execution_kind": "native_gpu",
            "arg_names": ["diagonal", "rhs"],
            "nvidia_krylov": self.contract,
        }


def package_nvidia_krylov_solver(
    *, shape: Sequence[int], storage: str = "f32", algorithm: str = "cg",
    tolerance: float = 1.0e-6, max_iterations: int = 128,
) -> NvidiaKrylovArtifact:
    contract = build_nvidia_krylov_contract(
        shape=shape, storage=storage, algorithm=algorithm,
        tolerance=tolerance, max_iterations=max_iterations,
    )
    return NvidiaKrylovArtifact(
        shape=tuple(int(dim) for dim in shape), contract=contract
    )


def build_nvidia_dense_krylov_contract(
    *, order: int, storage: str = "f32", algorithm: str = "gmres",
    tolerance: float = 1.0e-6, max_iterations: int = 128,
    restart: int = 16, reduction_ctas: int = 0,
) -> dict[str, Any]:
    """Build the cooperative-grid dense-operator Krylov contract.

    ``algorithm="cg"`` is admitted only under an authored SPD promise.  The
    CUDA kernel checks positive curvature at every iteration and fails closed
    when that promise is violated.  ``gmres`` carries no symmetry promise and
    uses twice-modified Gram-Schmidt plus a true-residual acceptance check.
    """
    n = int(order)
    if n <= 0 or n > 8192:
        raise ValueError("NVIDIA dense Krylov order must be in 1..8192")
    if storage not in {"f32", "f16", "bf16"}:
        raise ValueError("NVIDIA dense Krylov storage must be f32, f16, or bf16")
    if algorithm not in {"cg", "gmres"}:
        raise ValueError("NVIDIA dense Krylov algorithm must be cg or gmres")
    if not (0.0 < float(tolerance) < 1.0):
        raise ValueError("NVIDIA dense Krylov tolerance must be in (0, 1)")
    if int(max_iterations) <= 0:
        raise ValueError("NVIDIA dense Krylov max_iterations must be positive")
    if int(restart) <= 0 or int(restart) > 32:
        raise ValueError("NVIDIA dense Krylov restart must be in 1..32")
    if int(reduction_ctas) < 0:
        raise ValueError("NVIDIA dense Krylov reduction_ctas cannot be negative")
    body = {
        "schema": _NVIDIA_DENSE_KRYLOV_SCHEMA,
        "target": "nvidia_sm120",
        "architecture": "sm120",
        "operator": "arbitrary_dense_row_major_v1",
        "operator_shape": [n, n],
        "rhs_shape": [n],
        "operator_assumption": "symmetric_positive_definite" if algorithm == "cg" else "general_nonsingular",
        "algorithm": algorithm,
        "numeric_policy": {
            "storage": storage,
            "accum": "fp32",
            "matvec_math": "storage_quantized_fp32_fma",
            "residual_check": "fp32_true_residual",
        },
        "state": [
            "solution", "residual", "matvec", "basis", "hessenberg",
            "givens", "dot_partials", "convergence",
        ],
        "state_residency": "single_cooperative_launch_device_resident",
        "reduction": "deterministic_multi_cta_two_level",
        "requested_reduction_ctas": int(reduction_ctas),
        "true_residual_check": True,
        "orthogonalization": "twice_modified_gram_schmidt" if algorithm == "gmres" else "not_applicable",
        "tolerance": float(tolerance),
        "max_iterations": int(max_iterations),
        "restart": int(restart),
        "workgroup_size": 256,
        "performance_eligible": True,
    }
    return {**body, "artifact_hash": _digest(body)}


@dataclass(frozen=True)
class NvidiaDenseKrylovArtifact:
    order: int
    contract: dict[str, Any]

    @property
    def artifact_hash(self) -> str:
        return str(self.contract["artifact_hash"])

    def runtime_metadata(self) -> dict[str, Any]:
        return {
            "target": "nvidia_sm120",
            "compiler_path": "nvidia_dense_krylov_compiled",
            "executable": True,
            "execution_kind": "native_gpu",
            "arg_names": ["operator", "rhs"],
            "nvidia_dense_krylov": self.contract,
        }


def package_nvidia_dense_krylov_solver(
    *, order: int, storage: str = "f32", algorithm: str = "gmres",
    tolerance: float = 1.0e-6, max_iterations: int = 128,
    restart: int = 16, reduction_ctas: int = 0,
) -> NvidiaDenseKrylovArtifact:
    contract = build_nvidia_dense_krylov_contract(
        order=order, storage=storage, algorithm=algorithm,
        tolerance=tolerance, max_iterations=max_iterations, restart=restart,
        reduction_ctas=reduction_ctas,
    )
    return NvidiaDenseKrylovArtifact(order=int(order), contract=contract)


__all__ = [
    "ScheduledSolverIFTArtifact",
    "PhysicalGeneralSolverArtifact",
    "NvidiaKrylovArtifact",
    "NvidiaDenseKrylovArtifact",
    "build_solver_ift_contract",
    "build_general_solver_product_contract",
    "build_physical_general_solver_contract",
    "compile_physical_general_solver_from_graph",
    "diagonal_sqrt_ift_reference",
    "build_nvidia_krylov_contract",
    "build_nvidia_dense_krylov_contract",
    "lower_scheduled_solver_ift",
    "package_physical_general_solver",
    "package_nvidia_krylov_solver",
    "package_nvidia_dense_krylov_solver",
]
