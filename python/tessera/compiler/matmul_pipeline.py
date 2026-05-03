"""Narrow end-to-end CPU lowering path for supported @jit op graphs.

This module handles straight-line dataflow built from supported Tessera ops:

    @tessera.jit
    def f(...):
        y = tessera.ops.<supported_op>(...)
        return tessera.ops.<supported_op>(y)

It creates inspectable Schedule IR, Tile IR, and Target IR artifacts from the
existing Graph IR text and executes the operation on CPU via NumPy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from .graph_ir import GraphIRFunction, GraphIRModule, IROp
from .op_catalog import GRAPH_OP_TO_SPEC, LEGACY_GRAPH_OP_ALIASES, SUPPORTED_CPU_OPS, canonical_graph_op_name


MATMUL_OPS = {"tessera.matmul", "tessera.gemm"}
CONV2D_OPS = {"tessera.conv2d_nhwc", "tessera.conv2d"}
UNARY_OPS = {
    "tessera.layer_norm",
    "tessera.relu",
    "tessera.sigmoid",
    "tessera.sin",
    "tessera.gelu",
    "tessera.rmsnorm_safe",
}
REDUCTION_OPS = {"tessera.softmax", "tessera.softmax_safe"}
LAYOUT_OPS = {"tessera.transpose", "tessera.cast"}
STATEFUL_FUNCTIONAL_OPS = {"tessera.adam"}


@dataclass
class ReferenceKVCache:
    keys: list[Any] = field(default_factory=list)
    values: list[Any] = field(default_factory=list)

    def append(self, key: Any, value: Any) -> "ReferenceKVCache":
        self.keys.append(np.asarray(key))
        self.values.append(np.asarray(value))
        return self

    def prune(self, max_entries: Optional[int] = None) -> "ReferenceKVCache":
        if max_entries is not None:
            self.keys = self.keys[-max_entries:]
            self.values = self.values[-max_entries:]
        return self


@dataclass(frozen=True)
class LoweringArtifact:
    """Textual artifact for one compiler layer."""

    level: str
    text: str


@dataclass(frozen=True)
class JitDiagnostic:
    """Developer-facing diagnostic for JIT lowering decisions."""

    severity: str
    code: str
    message: str

    def format(self) -> str:
        return f"{self.severity.upper()}[{self.code}]: {self.message}"


@dataclass(frozen=True)
class CPUPlan:
    """Executable CPU plan for straight-line supported Graph IR ops."""

    function_name: str
    ops: tuple[IROp, ...]
    output_name: str
    tile: tuple[int, int, int]
    graph_ir: str
    schedule_ir: str
    tile_ir: str
    target_ir: str

    @property
    def op_name(self) -> str:
        return self.ops[-1].op_name

    @property
    def operand_names(self) -> tuple[str, ...]:
        names = []
        produced = {op.result for op in self.ops if op.result}
        for op in self.ops:
            for operand in op.operands:
                name = _operand_name(operand)
                if name not in produced and name not in names:
                    names.append(name)
        return tuple(names)

    def execute(self, args: Sequence[Any], kwargs: Mapping[str, Any], arg_names: Sequence[str]) -> Any:
        values = {name: value for name, value in zip(arg_names, args)}
        values.update(kwargs)
        for op in self.ops:
            operand_names = tuple(_operand_name(operand) for operand in op.operands)
            missing = [name for name in operand_names if name not in values]
            if missing:
                raise ValueError(f"CPU plan requires operand(s): {', '.join(missing)}")
            operands = [_as_value(values[name]) for name in operand_names]
            if op.result is None:
                raise ValueError(f"CPU plan cannot execute void op {op.op_name!r}")
            values[op.result] = _execute_op(op.op_name, operands, op.kwargs)
        if self.output_name not in values:
            raise ValueError(f"CPU plan did not produce output {self.output_name!r}")
        return values[self.output_name]

    def artifacts(self) -> tuple[LoweringArtifact, ...]:
        return (
            LoweringArtifact("graph", self.graph_ir),
            LoweringArtifact("schedule", self.schedule_ir),
            LoweringArtifact("tile", self.tile_ir),
            LoweringArtifact("target", self.target_ir),
        )


MatmulCPUPlan = CPUPlan


def build_matmul_cpu_plan(
    module: GraphIRModule,
    *,
    tile: tuple[int, int, int] = (128, 128, 64),
) -> Optional[CPUPlan]:
    """Backward-compatible alias for the general single-op CPU planner."""

    return build_cpu_plan(module, tile=tile)


def build_cpu_plan(
    module: GraphIRModule,
    *,
    tile: tuple[int, int, int] = (128, 128, 64),
) -> Optional[CPUPlan]:
    """Build a CPU plan if the Graph IR module is supported straight-line dataflow."""

    _validate_tile(tile)
    if len(module.functions) != 1:
        return None
    fn = module.functions[0]
    if not fn.body:
        return None
    for op in fn.body:
        if _canonical_op_name(op.op_name) not in SUPPORTED_CPU_OPS or not _valid_arity(op):
            return None
        operand_names = tuple(_operand_name(operand) for operand in op.operands)
        if any(not name or name == "?" for name in operand_names):
            return None
        if op.result is None:
            return None
    output_name = fn.body[-1].result
    if output_name is None:
        return None

    graph_text = module.to_mlir()
    ops = tuple(fn.body)
    schedule = _render_schedule_ir(fn, ops, tile=tile)
    tile_ir = _render_tile_ir(fn, ops)
    target = _render_target_ir(fn, ops)
    return CPUPlan(
        function_name=fn.name,
        ops=ops,
        output_name=output_name,
        tile=tile,
        graph_ir=graph_text,
        schedule_ir=schedule,
        tile_ir=tile_ir,
        target_ir=target,
    )


def explain_cpu_plan(module: GraphIRModule, *, target: str = "cpu") -> JitDiagnostic:
    """Return a diagnostic explaining compile-path or fallback status."""

    if target != "cpu":
        return JitDiagnostic(
            "warning",
            "JIT_EAGER_FALLBACK_TARGET",
            f"native target {target!r} execution is not wired; using eager Python fallback",
        )
    if not module.functions:
        return JitDiagnostic("warning", "JIT_EAGER_FALLBACK_EMPTY", "no Graph IR function was emitted")
    fn = module.functions[0]
    if not fn.body:
        return JitDiagnostic("warning", "JIT_EAGER_FALLBACK_EMPTY", "no Graph IR function body was emitted")
    unsupported = [op for op in fn.body if _canonical_op_name(op.op_name) not in SUPPORTED_CPU_OPS]
    if unsupported:
        names = ", ".join(sorted(SUPPORTED_CPU_OPS))
        seen = unsupported[0].op_name
        return JitDiagnostic(
            "warning",
            "JIT_EAGER_FALLBACK_UNSUPPORTED_OP",
            f"op {seen!r} is not supported by the CPU compiler path; supported ops: {names}",
        )
    bad_arity = [op for op in fn.body if not _valid_arity(op)]
    if bad_arity:
        return JitDiagnostic(
            "warning",
            "JIT_EAGER_FALLBACK_ARITY",
            f"op {bad_arity[0].op_name!r} has unsupported operand count",
        )
    unknown = [
        op
        for op in fn.body
        if op.result is None or any(_operand_name(operand) in {"", "?"} for operand in op.operands)
    ]
    if unknown:
        return JitDiagnostic(
            "warning",
            "JIT_EAGER_FALLBACK_UNSUPPORTED_BODY",
            "CPU compiler needs named values for every supported op; using eager Python fallback",
        )
    return JitDiagnostic(
        "info",
        "JIT_COMPILED_CPU",
        f"compiled {fn.name} through Graph IR -> Schedule IR -> Tile IR -> Target IR -> CPU",
    )


def _render_schedule_ir(
    fn: GraphIRFunction,
    ops: Sequence[IROp],
    *,
    tile: tuple[int, int, int],
) -> str:
    tile_m, tile_n, tile_k = tile
    lines = [
        'module attributes {tessera.ir.level = "schedule"} {',
        f'  "tessera.schedule.func"() ({{',
    ]
    for idx, op in enumerate(ops):
        operand_names = tuple(_operand_name(operand) for operand in op.operands)
        operand_attr = ", ".join(f'"{name}"' for name in operand_names)
        op_name = _canonical_op_name(op.op_name)
        if op_name in MATMUL_OPS:
            lines.append(
                f'    "schedule.tile"() {{source = "{op_name}", result = "{op.result}", ordinal = {idx} : i64, tile_m = {tile_m} : i64, tile_n = {tile_n} : i64, tile_k = {tile_k} : i64}} : () -> ()'
            )
        elif op_name in CONV2D_OPS:
            lines.append(
                f'    "schedule.tile"() {{source = "{op_name}", result = "{op.result}", ordinal = {idx} : i64, tile_h = 16 : i64, tile_w = 16 : i64, tile_c = 32 : i64}} : () -> ()'
            )
        elif op_name == "tessera.flash_attn":
            lines.append(
                f'    "schedule.pipeline.region"() ({{'
            )
            lines.append(
                f'      "schedule.stage"() ({{'
            )
            lines.append(
                f'        "schedule.prefetch"() {{source = "{op_name}", result = "{op.result}", ordinal = {idx} : i64, into = "shared", overlap = "compute", tile_q = 64 : i64, tile_kv = 64 : i64}} : () -> ()'
            )
            lines.append(
                f'        "schedule.yield"() : () -> ()'
            )
            lines.append(
                f'      }}) {{devices = [0]}} : () -> ()'
            )
            lines.append(
                f'      "schedule.yield"() : () -> ()'
            )
            lines.append(
                f'    }}) {{schedule = "fa4", micro_batches = 1 : i32, source = "{op_name}", result = "{op.result}", ordinal = {idx} : i64}} : () -> ()'
            )
        else:
            lines.append(
                f'    "schedule.elementwise"() {{source = "{op_name}", result = "{op.result}", ordinal = {idx} : i64, vectorize = true}} : () -> ()'
            )
        lines.append(
            f'    "schedule.layout"() {{operands = [{operand_attr}], layout = "row_major", ordinal = {idx} : i64}} : () -> ()'
        )
        if op_name.startswith("tessera.kv_cache."):
            lines.append(
                f'    "schedule.prefetch"() {{source = "{op_name}", result = "{op.result}", ordinal = {idx} : i64, into = "shared", overlap = "compute"}} : () -> ()'
            )
    lines.extend([
        f'  }}) {{sym_name = "{fn.name}", target = "cpu"}} : () -> ()',
        "}",
    ])
    return "\n".join(lines)


def _render_tile_ir(fn: GraphIRFunction, ops: Sequence[IROp]) -> str:
    lines = [
        'module attributes {tessera.ir.level = "tile"} {',
        f'  "tessera.tile.func"() ({{',
    ]
    for idx, op in enumerate(ops):
        op_name = _canonical_op_name(op.op_name)
        lines.append(
            f'    "{_tile_op_name(op_name)}"() {{source = "{op_name}", result = "{op.result}", ordinal = {idx} : i64, lowering = "{_lowering_kind(op_name)}", vectorize = true}} : () -> ()'
        )
        if op_name == "tessera.flash_attn":
            lines.append(
                f'    "tile.async_copy"() {{source = "{op_name}", result = "{op.result}", ordinal = {idx} : i64, stage = 0 : i32, vector = 16 : i32}} : () -> ()'
            )
            lines.append(
                f'    "tessera.attn.online_softmax"() {{source = "{op_name}", result = "{op.result}", ordinal = {idx} : i64, policy = "safe"}} : () -> ()'
            )
            lines.append(
                f'    "tile.wait_async"() {{stage = 0 : i32}} : () -> ()'
            )
        if op_name.startswith("tessera.kv_cache."):
            lines.append(
                f'    "tile.kv_cache"() {{source = "{op_name}", result = "{op.result}", ordinal = {idx} : i64, storage = "paged"}} : () -> ()'
            )
    lines.extend([
        f'  }}) {{sym_name = "{fn.name}", target = "cpu"}} : () -> ()',
        "}",
    ])
    return "\n".join(lines)


def _render_target_ir(fn: GraphIRFunction, ops: Sequence[IROp]) -> str:
    lines = [
        'module attributes {tessera.ir.level = "target", target = "cpu"} {',
        f'  "tessera.cpu.func"() ({{',
    ]
    for idx, op in enumerate(ops):
        op_name = _canonical_op_name(op.op_name)
        lines.append(
            f'    "{_target_op_name(op_name)}"() {{source = "{op_name}", result = "{op.result}", ordinal = {idx} : i64, abi = "numpy"}} : () -> ()'
        )
    lines.extend([
        f'  }}) {{sym_name = "{fn.name}"}} : () -> ()',
        "}",
    ])
    return "\n".join(lines)


def _operand_name(operand: str) -> str:
    return operand[1:] if operand.startswith("%") else operand


def _validate_tile(tile: tuple[int, int, int]) -> None:
    if len(tile) != 3 or any(int(v) <= 0 for v in tile):
        raise ValueError("CPU matmul tile must be a positive (tile_m, tile_n, tile_k) tuple")


def _valid_arity(op: IROp) -> bool:
    spec = GRAPH_OP_TO_SPEC.get(_canonical_op_name(op.op_name))
    return spec.valid_arity(len(op.operands)) if spec is not None else False


def _execute_op(op_name: str, operands: Sequence[np.ndarray], kwargs: Mapping[str, Any]) -> Any:
    op_name = _canonical_op_name(op_name)
    if op_name in MATMUL_OPS:
        return np.matmul(operands[0], operands[1])
    if op_name in CONV2D_OPS:
        bias = operands[2] if len(operands) > 2 else kwargs.get("bias", None)
        return _conv2d_nhwc(operands[0], operands[1], bias=bias, stride=kwargs.get("stride", 1), padding=kwargs.get("padding", 0))
    if op_name == "tessera.layer_norm":
        x = np.asarray(operands[0])
        eps = float(kwargs.get("eps", 1e-5))
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        return (x - mean) / np.sqrt(var + eps)
    if op_name == "tessera.relu":
        return np.maximum(0, operands[0])
    if op_name == "tessera.sigmoid":
        return 1.0 / (1.0 + np.exp(-operands[0]))
    if op_name == "tessera.sin":
        return np.sin(operands[0])
    if op_name == "tessera.gelu":
        x = np.asarray(operands[0])
        return x * 0.5 * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))
    if op_name in {"tessera.softmax", "tessera.softmax_safe"}:
        x = operands[0]
        axis = int(kwargs.get("axis", -1))
        e = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return e / np.sum(e, axis=axis, keepdims=True)
    if op_name == "tessera.rmsnorm_safe":
        x = np.asarray(operands[0])
        eps = float(kwargs.get("eps", 1e-6))
        return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)
    if op_name == "tessera.transpose":
        axes = kwargs.get("axes", None)
        if isinstance(axes, list):
            axes = tuple(axes)
        return np.transpose(operands[0], axes)
    if op_name == "tessera.cast":
        dtype = str(kwargs.get("dtype", "fp32"))
        dtype_map = {"bf16": np.float32, "fp16": np.float16, "fp32": np.float32, "fp64": np.float64}
        return np.asarray(operands[0]).astype(dtype_map.get(dtype, np.float32))
    if op_name == "tessera.dropout":
        x = np.asarray(operands[0])
        if not bool(kwargs.get("training", True)):
            return x
        p = float(kwargs.get("p", 0.1))
        if not 0.0 <= p < 1.0:
            raise ValueError("dropout p must be in [0.0, 1.0)")
        seed = kwargs.get("seed", None)
        rng = np.random.default_rng(None if seed is None else int(seed))
        mask = rng.binomial(1, 1.0 - p, x.shape) / (1.0 - p)
        return x * mask
    if op_name == "tessera.flash_attn":
        return _flash_attn_reference(operands[0], operands[1], operands[2], kwargs)
    if op_name in {"tessera.all_reduce", "tessera.reduce_scatter", "tessera.all_gather"}:
        return operands[0]
    if op_name == "tessera.fused_epilogue":
        x = np.asarray(operands[0])
        bias = operands[1] if len(operands) > 1 else kwargs.get("bias", None)
        if bias is not None:
            x = x + bias
        activation = kwargs.get("activation", "linear")
        if activation == "gelu":
            return _execute_op("tessera.gelu", [x], {})
        if activation == "relu":
            return np.maximum(0, x)
        return x
    if op_name == "tessera.fft":
        return np.fft.fft(operands[0], axis=int(kwargs.get("axis", -1)))
    if op_name == "tessera.ifft":
        return np.fft.ifft(operands[0], axis=int(kwargs.get("axis", -1)))
    if op_name == "tessera.rfft":
        return np.fft.rfft(operands[0], axis=int(kwargs.get("axis", -1)))
    if op_name == "tessera.irfft":
        n = kwargs.get("n", None)
        return np.fft.irfft(operands[0], n=None if n is None else int(n), axis=int(kwargs.get("axis", -1)))
    if op_name == "tessera.dct":
        return _dct_reference(operands[0], axis=int(kwargs.get("axis", -1)))
    if op_name == "tessera.spectral_conv":
        x = np.asarray(operands[0])
        w = np.asarray(operands[1])
        n = x.shape[-1] + w.shape[-1] - 1
        nfft = 1 << int(np.ceil(np.log2(n)))
        y = np.fft.irfft(np.fft.rfft(x, nfft) * np.fft.rfft(w, nfft), nfft)
        return y[..., :n]
    if op_name == "tessera.kv_cache.append":
        cache = operands[0] if isinstance(operands[0], ReferenceKVCache) else ReferenceKVCache()
        return cache.append(operands[1], operands[2])
    if op_name == "tessera.kv_cache.prune":
        cache = operands[0] if isinstance(operands[0], ReferenceKVCache) else ReferenceKVCache()
        max_entries = kwargs.get("max_entries", kwargs.get("max_seq", None))
        return cache.prune(None if max_entries is None else int(max_entries))
    if op_name == "tessera.adam":
        param, grad, moment1, moment2 = operands
        beta1 = float(kwargs.get("beta1", 0.9))
        beta2 = float(kwargs.get("beta2", 0.999))
        lr = float(kwargs.get("lr", 1e-3))
        eps = float(kwargs.get("eps", 1e-8))
        step = int(kwargs.get("step", 1))
        new_m = beta1 * moment1 + (1.0 - beta1) * grad
        new_v = beta2 * moment2 + (1.0 - beta2) * (grad * grad)
        m_hat = new_m / (1.0 - beta1**step)
        v_hat = new_v / (1.0 - beta2**step)
        new_param = param - lr * m_hat / (np.sqrt(v_hat) + eps)
        return new_param, new_m, new_v
    raise ValueError(f"unsupported CPU op {op_name!r}")


def _tile_op_name(op_name: str) -> str:
    op_name = _canonical_op_name(op_name)
    bare = op_name.split(".")[-1]
    if op_name in MATMUL_OPS:
        return "tile.mma"
    if op_name in CONV2D_OPS:
        return "tile.conv2d"
    return f"tile.{bare}"


def _target_op_name(op_name: str) -> str:
    op_name = _canonical_op_name(op_name)
    bare = op_name.split(".")[-1]
    if op_name in MATMUL_OPS:
        bare = "matmul"
    if op_name in CONV2D_OPS:
        bare = "conv2d_nhwc"
    return f"tessera.cpu.{bare}"


def _lowering_kind(op_name: str) -> str:
    spec = GRAPH_OP_TO_SPEC.get(_canonical_op_name(op_name))
    return spec.lowering if spec is not None else "elementwise"


def _canonical_op_name(op_name: str) -> str:
    return canonical_graph_op_name(LEGACY_GRAPH_OP_ALIASES.get(op_name, op_name))


def _as_value(value: Any) -> Any:
    if isinstance(value, ReferenceKVCache):
        return value
    if hasattr(value, "numpy") and callable(value.numpy):
        return np.asarray(value.numpy())
    if hasattr(value, "_data"):
        return np.asarray(value._data)
    return np.asarray(value)


def _pair(value: Any) -> tuple[int, int]:
    if isinstance(value, (tuple, list)):
        return int(value[0]), int(value[1])
    return int(value), int(value)


def _conv2d_nhwc(x: Any, weight: Any, *, bias: Any = None, stride: Any = 1, padding: Any = 0) -> np.ndarray:
    x = np.asarray(x)
    weight = np.asarray(weight)
    stride_h, stride_w = _pair(stride)
    pad_h, pad_w = _pair(padding)
    if x.ndim != 4 or weight.ndim != 4:
        raise ValueError("conv2d reference expects NHWC input and HWIO weights")
    x_pad = np.pad(x, ((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)))
    batch, in_h, in_w, _ = x_pad.shape
    k_h, k_w, _, out_c = weight.shape
    out_h = (in_h - k_h) // stride_h + 1
    out_w = (in_w - k_w) // stride_w + 1
    out = np.zeros((batch, out_h, out_w, out_c), dtype=np.result_type(x, weight))
    for i in range(out_h):
        for j in range(out_w):
            window = x_pad[:, i * stride_h:i * stride_h + k_h, j * stride_w:j * stride_w + k_w, :]
            out[:, i, j, :] = np.tensordot(window, weight, axes=([1, 2, 3], [0, 1, 2]))
    if bias is not None:
        out = out + np.asarray(bias)
    return out


def _flash_attn_reference(q: Any, k: Any, v: Any, kwargs: Mapping[str, Any]) -> np.ndarray:
    q = np.asarray(q)
    k = np.asarray(k)
    v = np.asarray(v)
    dropout_p = float(kwargs.get("dropout_p", 0.0))
    if not 0.0 <= dropout_p < 1.0:
        raise ValueError("dropout_p must be in [0.0, 1.0)")
    scale = kwargs.get("scale", None)
    scale = 1.0 / np.sqrt(q.shape[-1]) if scale is None else float(scale)
    scores = np.matmul(q, np.swapaxes(k, -1, -2)) * scale
    if bool(kwargs.get("causal", False)):
        q_len, k_len = scores.shape[-2], scores.shape[-1]
        mask = np.triu(np.ones((q_len, k_len), dtype=bool), k=1 + max(k_len - q_len, 0))
        scores = np.where(mask, -np.inf, scores)
    weights = _execute_op("tessera.softmax", [scores], {})
    if dropout_p > 0.0:
        seed = kwargs.get("seed", None)
        rng = np.random.default_rng(None if seed is None else int(seed))
        keep = rng.binomial(1, 1.0 - dropout_p, weights.shape)
        weights = weights * keep / (1.0 - dropout_p)
    return np.matmul(weights, v)


def _dct_reference(x: Any, axis: int = -1) -> np.ndarray:
    x = np.asarray(x)
    n = x.shape[axis]
    y = np.concatenate([x, np.flip(x, axis=axis)], axis=axis)
    spec = np.fft.fft(y, axis=axis)
    slicer = [slice(None)] * spec.ndim
    slicer[axis] = slice(0, n)
    return np.real(spec[tuple(slicer)])


__all__ = [
    "LoweringArtifact",
    "JitDiagnostic",
    "MATMUL_OPS",
    "CPUPlan",
    "MatmulCPUPlan",
    "ReferenceKVCache",
    "SUPPORTED_CPU_OPS",
    "build_cpu_plan",
    "build_matmul_cpu_plan",
    "explain_cpu_plan",
]
