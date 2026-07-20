"""X86-E2E-2 cohort 3/4 stable-ABI registry and explicit packaging.

This module describes exported C entries, not automatic Graph-IR promotion.
Callers must provide the ABI-shaped buffers (for example SDDMM's transposed
``Bt`` and Clifford's blade-major inputs).  Host-orchestrated public semantics
remain on their retained routes until a complete compiler-owned composition is
measured and selected.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any, Mapping, cast

from .native_artifact import (
    BufferBinding,
    LaunchDescriptor,
    LaunchGeometry,
    OrderingSemantics,
    ScalarArgument,
    ShapeGuard,
)
from .graph_ir import GraphIRModule
from .x86_native import X86NativePackage, _image, _lower


@dataclass(frozen=True)
class X86ABIArgument:
    name: str
    kind: str  # buffer | scalar
    dtype: str
    direction: str = "input"

    @property
    def mlir_type(self) -> str:
        if self.kind == "buffer":
            return "!llvm.ptr"
        return {"int64": "i64", "uint64": "i64", "int32": "i32", "uint32": "i32",
                "float32": "f32", "float64": "f64"}[self.dtype]


@dataclass(frozen=True)
class X86ABISpec:
    key: str
    cohort: int
    family: str
    symbol: str
    abi_id: str
    effects: str
    args: tuple[X86ABIArgument, ...]
    returns_status: bool = False
    public_route: str = "direct_abi"


def _b(name: str, dtype: str = "fp32", direction: str = "input") -> X86ABIArgument:
    return X86ABIArgument(name, "buffer", dtype, direction)


def _s(name: str, dtype: str = "int64") -> X86ABIArgument:
    return X86ABIArgument(name, "scalar", dtype)


def _spec(key: str, cohort: int, family: str, symbol: str, effects: str,
          *args: X86ABIArgument, status: bool = False,
          route: str = "direct_abi") -> X86ABISpec:
    abi = "tessera.x86." + key.replace("_", ".") + ".v1"
    return X86ABISpec(key, cohort, family, symbol, abi, effects, args, status, route)


_SPECS = (
    # Cohort 3: sparse, movement, spectral, linalg, and Clifford.
    _spec("spmm_csr_f32", 3, "sparse", "tessera_x86_avx512_spmm_csr_f32", "writeonly",
          _b("indptr", "int32"), _b("indices", "int32"), _b("values"), _b("rhs"),
          _s("M"), _s("N"), _b("output", direction="output"), route="abi_shaped"),
    _spec("sddmm_f32", 3, "sparse", "tessera_x86_avx512_sddmm_f32", "writeonly",
          _b("lhs"), _b("rhs_transposed"), _b("mask"), _s("M"), _s("N"), _s("K"),
          _b("output", direction="output"), route="host_packed"),
    _spec("gather_f32", 3, "movement", "tessera_x86_gather_f32", "writeonly",
          _b("source"), _s("SourceN"), _b("indices", "int64"), _s("N"),
          _b("output", direction="output")),
    _spec("scatter_f32", 3, "movement", "tessera_x86_scatter_f32", "readwrite",
          _b("output", direction="inout"), _s("OutputRows"), _b("source"),
          _b("indices", "int64"), _s("IndexCount"), _s("RowLength"), _s("Mode", "int32")),
    _spec("bitonic_sort_kv_f32", 3, "sort", "tessera_x86_bitonic_sort_kv_f32", "readwrite",
          _b("keys", direction="inout"), _b("indices", "int64", "inout"),
          _s("N"), _s("Descending", "int32"), route="host_padded"),
    _spec("fft_c2c_f32", 3, "fft", "tessera_x86_fft_c2c_f32", "readwrite",
          _b("interleaved_complex", direction="inout"), _s("Batch"), _s("N"),
          _s("Inverse", "int32"), route="radix2_direct"),
    _spec("cholesky_f32", 3, "linalg", "tessera_x86_cholesky_f32", "writeonly",
          _b("matrix"), _s("Batch"), _s("N"), _b("lower", direction="output")),
    _spec("tri_solve_f32", 3, "linalg", "tessera_x86_tri_solve_f32", "writeonly",
          _b("matrix"), _b("rhs"), _s("Batch"), _s("N"), _s("M"),
          _s("Lower", "int32"), _b("output", direction="output")),
    _spec("lu_f32", 3, "linalg", "tessera_x86_lu_f32", "writeonly",
          _b("matrix"), _s("Batch"), _s("N"), _b("lu", direction="output"),
          _b("pivots", "int32", "output")),
    _spec("qr_f32", 3, "linalg", "tessera_x86_qr_f32", "writeonly",
          _b("matrix"), _s("Batch"), _s("M"), _s("N"),
          _b("q", direction="output"), _b("r", direction="output")),
    _spec("svd_f32", 3, "linalg", "tessera_x86_svd_f32", "writeonly",
          _b("matrix"), _s("Batch"), _s("M"), _s("N"),
          _b("u", direction="output"), _b("s", direction="output"),
          _b("vt", direction="output")),
    _spec("clifford_bilinear_f32", 3, "clifford", "tessera_x86_clifford_bilinear_f32", "writeonly",
          _b("lhs_blade_major"), _b("rhs_blade_major"), _s("N"),
          _s("Kind", "int32"), _b("output_blade_major", direction="output"), route="host_packed"),

    # Cohort 4: loss, quantization, cache/state, MoE, optimizer, RNG, EBM.
    _spec("pointwise_loss_f32", 4, "loss", "tessera_x86_avx512_pointwise_loss_f32", "writeonly",
          _b("prediction"), _b("target"), _s("N"), _s("Kind", "int32"),
          _s("Parameter", "float32"), _b("output", direction="output"), route="per_element"),
    _spec("binary_loss_f32", 4, "loss", "tessera_x86_avx512_binary_loss_f32", "writeonly",
          _b("logits"), _b("target"), _s("N"), _s("Kind", "int32"),
          _s("PositiveWeight", "float32"), _s("NegativeWeight", "float32"),
          _b("output", direction="output"), route="per_element"),
    _spec("policy_loss_f32", 4, "loss", "tessera_x86_avx512_policy_loss_f32", "writeonly",
          _b("logp_new"), _b("logp_old"), _b("advantage"), _s("N"),
          _s("Kind", "int32"), _s("Clip", "float32"), _b("output", direction="output"), route="per_element"),
    _spec("fpquant_f32", 4, "quantization", "tessera_x86_avx512_fpquant_f32", "writeonly",
          _b("source"), _s("N"), _s("MaxNormal", "float32"), _s("MantissaBits", "int32"),
          _s("MinExponent", "int32"), _b("output", direction="output")),
    _spec("selective_ssm_f32", 4, "ssm", "tessera_x86_avx512_selective_ssm_f32", "readwrite",
          _b("x"), _b("a2d"), _b("b"), _b("c"), _b("delta"),
          _s("Batch"), _s("Sequence"), _s("D"), _s("N"),
          _b("state", direction="inout"), _b("output", direction="output")),
    _spec("selective_ssm_f16", 4, "ssm", "tessera_x86_avx512_selective_ssm_f16", "readwrite",
          _b("x", "fp16"), _b("a2d", "fp16"), _b("b", "fp16"), _b("c", "fp16"),
          _b("delta", "fp16"), _s("Batch"), _s("Sequence"), _s("D"), _s("N"),
          _b("state", direction="inout"), _b("output", "fp16", "output")),
    _spec("selective_ssm_bf16", 4, "ssm", "tessera_x86_avx512_selective_ssm_bf16", "readwrite",
          _b("x", "bf16"), _b("a2d", "bf16"), _b("b", "bf16"), _b("c", "bf16"),
          _b("delta", "bf16"), _s("Batch"), _s("Sequence"), _s("D"), _s("N"),
          _b("state", direction="inout"), _b("output", "bf16", "output")),
    _spec("selective_ssm_bwd_f32", 4, "ssm", "tessera_x86_selective_ssm_bwd_f32", "writeonly",
          *tuple(_b(name) for name in ("x", "a2d", "b", "c", "delta", "dy")),
          _s("Batch"), _s("Sequence"), _s("D"), _s("N"),
          *tuple(_b(name, direction="output") for name in ("state_trajectory", "dx", "da2d", "db", "dc", "ddelta"))),
    _spec("moe_f32", 4, "moe", "tessera_x86_moe_f32", "writeonly",
          _b("tokens"), _b("experts"), _b("routes", "int32"), _s("TokenCount"),
          _s("InputDim"), _s("OutputDim"), _b("output", direction="output")),
    _spec("optimizer_f32", 4, "optimizer", "tessera_x86_optimizer_f32", "writeonly",
          *tuple(_b(name) for name in ("parameters", "gradients", "momentum", "variance")),
          _s("N"), _s("Kind", "int32"),
          *tuple(_s(name, "float32") for name in ("LearningRate", "Beta1", "Beta2", "Epsilon", "WeightDecay", "Beta1Correction", "Beta2Correction")),
          *tuple(_b(name, direction="output") for name in ("parameters_out", "momentum_out", "variance_out"))),
    _spec("deltanet_f32", 4, "deltanet", "tessera_x86_deltanet_f32", "writeonly",
          *tuple(_b(name) for name in ("q", "k", "v", "gate", "beta", "decay")),
          _b("output", direction="output"), _s("Batch"), _s("Heads"), _s("Sequence"),
          _s("Dqk"), _s("Dv"), *tuple(_s(name, "int32") for name in
          ("Erase", "Modified", "HasGate", "HasBeta", "HasDecay"))),
    _spec("kv_cache_append_f32", 4, "kv_cache", "tessera_x86_kv_cache_append_f32", "stateful",
          _b("cache", direction="inout"), _s("MaxSequence"), _s("RowLength"), _s("Start"),
          _b("rows"), _s("RowCount"), status=True),
    _spec("kv_cache_read_f32", 4, "kv_cache", "tessera_x86_kv_cache_read_f32", "stateful",
          _b("cache"), _s("MaxSequence"), _s("RowLength"), _s("Start"), _s("End"),
          _b("output", direction="output"), status=True),
    _spec("kv_cache_prune_f32", 4, "kv_cache", "tessera_x86_kv_cache_prune_f32", "stateful",
          _b("cache", direction="inout"), _s("MaxSequence"), _s("RowLength"),
          _s("CurrentSequence"), _s("Limit"), status=True),
    _spec("philox_uniform_f32", 4, "rng", "tessera_x86_philox_uniform_f32", "writeonly",
          _s("Seed", "uint64"), _s("CounterBase", "uint64"), _s("N"),
          _b("output", direction="output"), route="uniform_core"),
    _spec("ebm_affine_langevin_f32", 4, "ebm", "tessera_x86_ebm_affine_langevin_f32", "writeonly",
          _b("y"), _b("gradient"), _b("noise"), _s("N"), _s("Eta", "float32"),
          _s("NoiseScale", "float32"), _b("output", direction="output")),
    _spec("ebm_decode_init_noise_apply_f32", 4, "ebm", "tessera_x86_ebm_decode_init_noise_apply_f32", "writeonly",
          _b("base"), _b("noise"), _s("N"), _s("Stddev", "float32"),
          _b("output", direction="output")),
    _spec("ebm_ebt_tiny_f32", 4, "ebm", "tessera_x86_ebm_ebt_tiny_f32", "writeonly",
          _b("y0"), _b("gradient"), _s("Eta", "float32"), _s("Steps", "int32"),
          _s("Batch"), _s("K"), _s("D"), _b("output", direction="output")),
    _spec("ebm_energy_quadratic_f32", 4, "ebm", "tessera_x86_ebm_energy_quadratic_f32", "writeonly",
          _b("x"), _b("y"), _s("Batch"), _s("D"), _b("output", direction="output")),
    _spec("ebm_langevin_philox_f32", 4, "ebm", "tessera_x86_ebm_langevin_philox_f32", "writeonly",
          _b("y"), _b("gradient"), _s("N"), _s("Eta", "float32"),
          _s("NoiseScale", "float32"), *tuple(_s(name, "uint32") for name in
          ("Key0", "Key1", "Counter0", "Counter1", "Counter2", "Counter3")),
          _b("output", direction="output")),
    _spec("ebm_partition_exact_f32", 4, "ebm", "tessera_x86_ebm_partition_exact_f32", "writeonly",
          _b("energies"), _s("N"), _s("Temperature", "float32"),
          _b("output", direction="output")),
)

X86_BREADTH_ABIS: Mapping[str, X86ABISpec] = {spec.key: spec for spec in _SPECS}

# Filled from the committed serial-alternating retained/descriptor corpus.
# ``None`` means the measured rows did not justify automatic promotion.
GRAPH_PROMOTION_THRESHOLDS: Mapping[str, int | None] = {
    "gather": 1_048_576,
    "pointwise_loss": 16_384,
    "cholesky": 2_048,
    "tri_solve": 512,
}


def cohort_specs(cohort: int) -> tuple[X86ABISpec, ...]:
    return tuple(spec for spec in _SPECS if spec.cohort == cohort)


def emit_abi_tile_ir(spec: X86ABISpec, *, entry: str) -> str:
    declarations = ", ".join(
        f"%a{index}: {argument.mlir_type}" for index, argument in enumerate(spec.args)
    )
    operands = ", ".join(f"%a{index}" for index in range(len(spec.args)))
    types = ", ".join(argument.mlir_type for argument in spec.args)
    status = str(spec.returns_status).lower()
    return f'''module {{
  llvm.func @{entry}({declarations}) {{
    tile.x86_abi_kernel {operands} {{
      symbol = "{spec.symbol}", abi = "{spec.abi_id}",
      family = "{spec.family}", effects = "{spec.effects}",
      returns_status = {status}
    }} : {types}
    llvm.return
  }}
}}
'''


def package_abi(
    key: str, *, pipeline_name: str, buffer_shapes: Mapping[str, tuple[int, ...]],
    buffer_names: Mapping[str, str] | None = None,
) -> X86NativePackage:
    """Package one explicit ABI-shaped call without changing selectors."""
    spec = X86_BREADTH_ABIS[key]
    public_names = dict(buffer_names or {})
    buffer_args = tuple(argument for argument in spec.args if argument.kind == "buffer")
    missing = {argument.name for argument in buffer_args} - set(buffer_shapes)
    extra = set(buffer_shapes) - {argument.name for argument in buffer_args}
    if missing or extra:
        raise ValueError(f"{key} buffer shape mismatch: missing={sorted(missing)}, extra={sorted(extra)}")
    tile_ir = emit_abi_tile_ir(spec, entry=f"tessera_tile_x86_{key}")
    target_ir, payload, compiler, toolchain = _lower(tile_ir, spec.symbol)
    image = _image(
        target_ir=target_ir, payload=payload, compiler=compiler, toolchain=toolchain,
        pipeline_name=pipeline_name, symbol=spec.symbol, abi=spec.abi_id,
    )
    buffers = tuple(
        BufferBinding(index, public_names.get(argument.name, argument.name),
                      argument.direction, argument.dtype,
                      len(buffer_shapes[argument.name]), "row_major", 4)
        for index, argument in enumerate(buffer_args)
    )
    scalars = tuple(
        ScalarArgument(len(buffers) + index, argument.name, argument.dtype)
        for index, argument in enumerate(arg for arg in spec.args if arg.kind == "scalar")
    )
    guards = tuple(
        ShapeGuard(public_names.get(name, name), axis, "eq", extent)
        for name, shape in buffer_shapes.items()
        for axis, extent in enumerate(shape)
        if extent > 0
    )
    descriptor = LaunchDescriptor(
        image_digest=image.image_digest, entry_symbol=spec.symbol, abi_id=spec.abi_id,
        buffers=buffers, scalars=scalars, shape_guards=guards,
        geometry=LaunchGeometry(policy=f"x86_avx512_{spec.family}"),
        ordering=OrderingSemantics(
            ordered_submission=True, residency="all", synchronization=("return",),
        ),
        provenance={
            "work_item": "X86-E2E-2", "cohort": spec.cohort,
            "family": spec.family, "effects": spec.effects,
            "public_route": spec.public_route, "returns_status": spec.returns_status,
            "abi_arguments": [
                {"name": arg.name, "kind": arg.kind, "dtype": arg.dtype,
                 "direction": arg.direction,
                 **({"binding": public_names[arg.name]} if arg.name in public_names else {})}
                for arg in spec.args
            ],
        },
    )
    return X86NativePackage(tile_ir, target_ir, target_ir, image, descriptor)


_POINTWISE_LOSSES: Mapping[str, tuple[int, str | None]] = {
    "tessera.mse_loss": (0, None), "tessera.loss.mse": (0, None),
    "tessera.mae_loss": (1, None), "tessera.loss.mae": (1, None),
    "tessera.huber_loss": (2, "delta"), "tessera.loss.huber": (2, "delta"),
    "tessera.smooth_l1_loss": (3, "beta"),
    "tessera.loss.smooth_l1": (3, "beta"),
    "tessera.log_cosh_loss": (4, None), "tessera.loss.log_cosh": (4, None),
}


def _static_shape(module: GraphIRModule, name: str) -> tuple[int, ...] | None:
    argument = next(
        (item for item in module.functions[0].args if item.name == name), None
    )
    if argument is None or argument.ir_type.rank is None:
        return None
    try:
        shape = tuple(int(value) for value in argument.ir_type.shape)
    except (TypeError, ValueError):
        return None
    return shape if shape and all(value > 0 for value in shape) else None


def _result_shape(module: GraphIRModule) -> tuple[int, ...] | None:
    if len(module.functions[0].result_types) != 1:
        return None
    try:
        shape = tuple(int(value) for value in module.functions[0].result_types[0].shape)
    except (TypeError, ValueError):
        return None
    return shape if all(value > 0 for value in shape) else None


def graph_breadth_contract(module: GraphIRModule) -> dict[str, Any] | None:
    """Return an isomorphic public-Graph/direct-ABI contract, if one exists."""
    if len(module.functions) != 1 or len(module.functions[0].body) != 1:
        return None
    function, op = module.functions[0], module.functions[0].body[0]
    args = {argument.name: argument for argument in function.args}
    names = tuple(value.removeprefix("%") for value in op.operands)
    output_name = op.result or "output"
    output_shape = _result_shape(module)
    if output_shape is None or function.result_types[0].dtype != "fp32":
        return None
    if op.op_name == "tessera.gather":
        if len(names) != 2 or any(name not in args for name in names):
            return None
        source_shape, index_shape = (_static_shape(module, name) for name in names)
        axis = op.kwargs.get("axis", 0)
        if (
            source_shape is None or index_shape is None
            or len(source_shape) != 1 or len(index_shape) != 1
            or output_shape != index_shape or axis not in {0, -1}
            or args[names[0]].ir_type.dtype != "fp32"
            or args[names[1]].ir_type.dtype != "int64"
        ):
            return None
        return {
            "key": "gather_f32", "family": "gather", "inputs": names,
            "output": output_name,
            "shapes": {"source": source_shape, "indices": index_shape,
                       "output": output_shape},
            "names": {"source": names[0], "indices": names[1],
                      "output": output_name},
            "scalars": {"SourceN": source_shape[0], "N": index_shape[0]},
        }
    if op.op_name in _POINTWISE_LOSSES:
        if len(names) != 2 or any(name not in args for name in names):
            return None
        shapes = tuple(_static_shape(module, name) for name in names)
        if (
            any(shape is None for shape in shapes) or len(set(shapes)) != 1
            or output_shape != shapes[0]
            or any(args[name].ir_type.dtype != "fp32" for name in names)
            or str(op.kwargs.get("reduction", "mean")) != "none"
        ):
            return None
        shape = shapes[0]
        kind, parameter_name = _POINTWISE_LOSSES[op.op_name]
        parameter = float(op.kwargs.get(parameter_name, 1.0)) if parameter_name else 0.0
        if not math.isfinite(parameter) or parameter < 0.0:
            return None
        return {
            "key": "pointwise_loss_f32", "family": "pointwise_loss",
            "inputs": names, "output": output_name,
            "shapes": {"prediction": shape, "target": shape, "output": output_shape},
            "names": {"prediction": names[0], "target": names[1],
                      "output": output_name},
            "scalars": {"N": math.prod(shape), "Kind": kind,
                        "Parameter": parameter},
            "kind": kind, "parameter": parameter,
        }
    if op.op_name in {"tessera.cholesky", "tessera.tri_solve"}:
        expected = 1 if op.op_name == "tessera.cholesky" else 2
        if len(names) != expected or any(name not in args for name in names):
            return None
        shapes = tuple(_static_shape(module, name) for name in names)
        matrix_shape = shapes[0]
        if (
            matrix_shape is None or len(matrix_shape) not in {2, 3}
            or matrix_shape[-1] != matrix_shape[-2]
            or any(args[name].ir_type.dtype != "fp32" for name in names)
        ):
            return None
        batch, n = (1, matrix_shape[0]) if len(matrix_shape) == 2 else (
            matrix_shape[0], matrix_shape[1]
        )
        if op.op_name == "tessera.cholesky":
            if output_shape != matrix_shape:
                return None
            return {
                "key": "cholesky_f32", "family": "cholesky",
                "inputs": names, "output": output_name,
                "shapes": {"matrix": matrix_shape, "lower": output_shape},
                "names": {"matrix": names[0], "lower": output_name},
                "scalars": {"Batch": batch, "N": n},
            }
        rhs_shape = shapes[1]
        if rhs_shape is None or output_shape != rhs_shape:
            return None
        if len(matrix_shape) == 2:
            valid_rhs = len(rhs_shape) == 2 and rhs_shape[0] == n
            columns = rhs_shape[1] if valid_rhs else 0
        else:
            valid_rhs = len(rhs_shape) == 3 and rhs_shape[:2] == (batch, n)
            columns = rhs_shape[2] if valid_rhs else 0
        if not valid_rhs:
            return None
        return {
            "key": "tri_solve_f32", "family": "tri_solve",
            "inputs": names, "output": output_name,
            "shapes": {"matrix": matrix_shape, "rhs": rhs_shape,
                       "output": output_shape},
            "names": {"matrix": names[0], "rhs": names[1],
                      "output": output_name},
            "scalars": {"Batch": batch, "N": n, "M": columns,
                        "Lower": int(bool(op.kwargs.get("lower", True)))},
        }
    return None


def requests_graph_breadth(module: GraphIRModule) -> bool:
    if len(module.functions) != 1 or len(module.functions[0].body) != 1:
        return False
    return module.functions[0].body[0].op_name in {
        "tessera.gather", "tessera.cholesky", "tessera.tri_solve",
        *_POINTWISE_LOSSES,
    }


def supports_graph_breadth(module: GraphIRModule) -> bool:
    return graph_breadth_contract(module) is not None


def supports_promoted_graph_breadth(module: GraphIRModule) -> bool:
    """Measured Level-C selector policy; thresholds are benchmark-owned."""
    contract = graph_breadth_contract(module)
    if contract is None:
        return False
    output_shape = _result_shape(module)
    assert output_shape is not None
    elements = math.prod(output_shape)
    family = str(contract["family"])
    threshold = GRAPH_PROMOTION_THRESHOLDS[family]
    return threshold is not None and elements >= threshold


def package_graph_breadth(
    module: GraphIRModule, *, pipeline_name: str,
) -> X86NativePackage:
    contract = graph_breadth_contract(module)
    if contract is None:
        raise ValueError("x86 breadth packaging requires one isomorphic static Graph operation")
    package = package_abi(
        str(contract["key"]), pipeline_name=pipeline_name,
        buffer_shapes=cast(Mapping[str, tuple[int, ...]], contract["shapes"]),
        buffer_names=cast(Mapping[str, str], contract["names"]),
    )
    descriptor = replace(package.descriptor, provenance={
        **package.descriptor.provenance,
        "graph_level": True, "selector_family": str(contract["family"]),
        "graph_scalars": cast(Mapping[str, object], contract["scalars"]),
        **({"kind": int(contract["kind"]),
            "parameter": float(contract["parameter"])}
           if "kind" in contract else {}),
    })
    return replace(package, descriptor=descriptor)
