"""Canonical tensor-valued attention VJP through Schedule -> launch Tile."""

from __future__ import annotations

import math
import hashlib
import re
from dataclasses import dataclass

from .attention_contract import plan_attention_backward_workspace
from .graph_ir import GraphIRModule
from .rocm_native import emit_attention_backward_graph_ir
from .scheduled_matmul import digest_text, find_tessera_opt, run_tessera_opt


_RECURRENCE = "tensor_dq_split_dkdv_fixed_reduce_v1"
_BACKWARD_HASH_RE = re.compile(
    r"tile\.attention_backward_kernel[\s\S]*?tessera\.schedule_hash = \"([0-9a-f]{64})\""
)


@dataclass(frozen=True)
class ScheduledAttentionBackwardArtifact:
    graph_ir: str
    schedule_ir: str
    semantic_ir: str
    tile_ir: str
    target: str
    architecture: str
    function_name: str
    input_names: tuple[str, str, str, str]
    bias_name: str | None
    output_names: tuple[str, str, str]
    dtype: str
    storage: str
    dims: tuple[int, int, int, int, int, int, int]
    scale: float
    causal: bool
    window_left: int
    window_right: int
    softcap: float
    dropout_p: float
    dropout_seed: int
    query_block: int
    key_block: int
    split_count: int
    reduction_order: tuple[int, ...]
    workspace_bytes: int
    workgroup_size: int
    recurrence: str
    lse_checkpoint_policy: str
    lse_checkpoint_selection: str
    schedule_digest: str

    @property
    def graph_digest(self) -> str:
        return digest_text(self.graph_ir)

    @property
    def schedule_ir_digest(self) -> str:
        return digest_text(self.schedule_ir)

    @property
    def semantic_ir_digest(self) -> str:
        return digest_text(self.semantic_ir)

    @property
    def tile_digest(self) -> str:
        return digest_text(self.tile_ir)

    def validate(self) -> None:
        if self.schedule_ir.count("schedule.attention_backward") != 1:
            raise ValueError("attention backward requires one three-result Schedule edge")
        if self.schedule_ir.count(self.schedule_digest) != 3:
            raise ValueError("attention backward has incomplete content identity")
        if self.tile_ir.count("tile.attention_backward_kernel") != 1:
            raise ValueError("attention backward requires one launch-level Tile op")
        if "tessera_attn.backward" in self.tile_ir or "schedule." in self.tile_ir:
            raise ValueError("attention backward Tile program retains Graph or Schedule IR")
        hashes = _BACKWARD_HASH_RE.findall(self.tile_ir)
        if hashes != [self.schedule_digest]:
            raise ValueError("attention backward Tile program has stale content identity")
        for marker in (
            'tessera.attention_backward_phase = "dq.key_block"',
            'tessera.attention_backward_phase = "dkdv.key_block"',
            'tessera.attention_backward_phase = "reduce.fixed_order_split"',
            "tessera_attn.backward_dq_block",
            "tessera_attn.backward_dkdv_block",
            "tessera_attn.backward_reduce_split",
        ):
            if marker not in self.semantic_ir:
                raise ValueError("attention backward lost a canonical tensor loop")
        for name in self.output_names:
            if not name:
                raise ValueError("attention backward output identity must be explicit")
        required = {
            "workspace_bytes": self.workspace_bytes,
            "split_count": self.split_count,
            "query_block": self.query_block,
            "key_block": self.key_block,
        }
        for name, value in required.items():
            if not re.search(rf"{name} = {value} : i64", self.tile_ir):
                raise ValueError(f"attention backward has stale {name}")
        if "reduction_order = array<i64: 0, 1>" not in self.tile_ir:
            raise ValueError("attention backward lost fixed reduction order")
        for name, value in (
            ("lse_checkpoint", self.lse_checkpoint_selection),
            ("tessera.lse_checkpoint_policy", self.lse_checkpoint_policy),
            ("tessera.attention_backward_recurrence", self.recurrence),
        ):
            if f'{name} = "{value}"' not in self.tile_ir:
                raise ValueError(f"attention backward has stale {name}")


def supports_scheduled_attention_backward(
    module: GraphIRModule, *, target: str
) -> bool:
    try:
        _graph_contract(module, target)
    except ValueError:
        return False
    return True


def lower_scheduled_attention_backward(
    module: GraphIRModule, *, target: str
) -> ScheduledAttentionBackwardArtifact:
    contract = _graph_contract(module, target)
    tool = find_tessera_opt()
    if tool is None:
        raise RuntimeError("scheduled attention backward requires production tessera-opt")
    (
        compiler_target,
        architecture,
        function_name,
        names,
        bias_name,
        result_names,
        dtype,
        storage,
        dims,
        scale,
        causal,
        window_left,
        window_right,
        softcap,
        dropout_p,
        dropout_seed,
        checkpoint_policy,
        checkpoint_selection,
    ) = contract
    semantic_key = hashlib.sha256(
        f"{scale:.17g}:{causal}:{bool(bias_name)}:{window_left}:"
        f"{window_right}:{softcap:.17g}:{dropout_p:.17g}:"
        f"{dropout_seed}:split_reduced:{checkpoint_selection}".encode()
    ).hexdigest()[:10]
    backward_entry = f"tessera_tile_attention_backward_{storage}_{semantic_key}"
    graph_ir = emit_attention_backward_graph_ir(
        forward_entry=f"tessera_tile_attention_bwd_recompute_{storage}_{semantic_key}",
        backward_entry=backward_entry,
        storage=storage,
        dims=dims,
        scale=scale,
        causal=causal,
        bias=bias_name is not None,
        window_left=window_left,
        window_right=window_right,
        softcap=softcap,
        dropout_p=dropout_p,
        dropout_seed=dropout_seed,
        query_block=16,
        key_block=16,
        split_count=2,
        save_lse=checkpoint_selection == "saved",
    ).replace(
        "module {",
        f'module attributes {{tessera.target = "{compiler_target}", '
        f'tessera.arch = "{architecture}"}} {{',
        1,
    )
    schedule_ir = run_tessera_opt(tool, graph_ir, "--tessera-graph-to-schedule")
    tile_ir = run_tessera_opt(tool, schedule_ir, "--tessera-schedule-to-tile")
    semantic_ir = run_tessera_opt(
        tool,
        graph_ir,
        f"--tessera-tile-ir-lowering=tile-q={dims[3]} tile-kv=16 sm=90",
    )
    hashes = _BACKWARD_HASH_RE.findall(tile_ir)
    if len(hashes) != 1:
        raise RuntimeError("attention backward lowering lost its program identity")
    workspace = plan_attention_backward_workspace(
        batch=dims[0], query_heads=dims[1], kv_heads=dims[2],
        query_rows=dims[3], key_rows=dims[4], head_dim=dims[5],
        value_dim=dims[6], split_count=2,
    )
    artifact = ScheduledAttentionBackwardArtifact(
        graph_ir, schedule_ir, semantic_ir, tile_ir, compiler_target,
        architecture, function_name, names, bias_name, result_names, dtype,
        storage, dims, scale, causal, window_left, window_right, softcap,
        dropout_p, dropout_seed, 16, 16, 2, (0, 1), workspace.bytes,
        256 if compiler_target == "rocm" else 1, _RECURRENCE,
        checkpoint_policy, checkpoint_selection, hashes[0],
    )
    artifact.validate()
    return artifact


def _shape(module: GraphIRModule, name: str) -> tuple[int, ...]:
    for argument in module.functions[0].args:
        if argument.name == name:
            try:
                return tuple(int(value) for value in argument.ir_type.shape)
            except (TypeError, ValueError) as error:
                raise ValueError("attention backward requires static shapes") from error
    raise ValueError(f"unknown attention backward operand {name!r}")


def _graph_contract(module: GraphIRModule, target: str) -> tuple:
    if target not in {"x86", "rocm_gfx1151"}:
        raise ValueError(
            "scheduled attention backward supports only x86 and rocm_gfx1151; "
            "gfx1200/gfx1250 require architecture-owned profiles and evidence"
        )
    if len(module.functions) != 1 or len(module.functions[0].body) != 1:
        raise ValueError("attention backward requires one Graph operation")
    function = module.functions[0]
    op = function.body[0]
    if op.op_name not in {"tessera.flash_attn_bwd", "tessera.flash_attn_vjp"}:
        raise ValueError("attention backward requires the canonical Graph VJP")
    if len(op.operands) not in {4, 5} or len(function.result_types) != 3:
        raise ValueError("attention backward requires four inputs and three outputs")
    names = tuple(value.removeprefix("%") for value in op.operands[:4])
    bias_name = op.operands[4].removeprefix("%") if len(op.operands) == 5 else None
    result_names = tuple(op.result_names)
    if len(result_names) != 3:
        raise ValueError("attention backward requires named dQ/dK/dV results")
    do_shape, q_shape, k_shape, v_shape = (_shape(module, name) for name in names)
    if any(len(shape) != 4 for shape in (do_shape, q_shape, k_shape, v_shape)):
        raise ValueError("attention backward requires rank-4 tensors")
    b, hq, sq, d = q_shape
    bk, hkv, sk, kd = k_shape
    bv, vh, vs, dv = v_shape
    if (bk, bv, vh, vs, kd, do_shape) != (b, b, hkv, sk, d, (b, hq, sq, dv)):
        raise ValueError("attention backward shapes or head mapping are inconsistent")
    if min(b, hq, hkv, sq, sk, d, dv) <= 0 or hq % hkv or d != dv:
        raise ValueError("attention backward requires positive compatible MHA/GQA/MQA shapes")
    args = {arg.name: arg for arg in function.args}
    dtypes = {args[name].ir_type.dtype for name in names}
    if len(dtypes) != 1:
        raise ValueError("attention backward inputs require one storage type")
    dtype = dtypes.pop()
    if target == "x86" and dtype != "fp32":
        raise ValueError("Zen 5 attention backward requires fp32")
    if target == "rocm_gfx1151" and (dtype not in {"fp16", "bf16"} or d % 16):
        raise ValueError("gfx1151 attention backward requires fp16/bf16 D%16=0")
    expected = (q_shape, k_shape, v_shape)
    for result, shape in zip(function.result_types, expected, strict=True):
        if result.dtype != "fp32" or tuple(map(int, result.shape)) != shape:
            raise ValueError("attention backward requires fp32 dQ/dK/dV identity")
    if bias_name is not None:
        if bias_name not in args or args[bias_name].ir_type.dtype != "fp32" or _shape(module, bias_name) != (b, hq, sq, sk):
            raise ValueError("attention backward bias contract is invalid")
    window = op.kwargs.get("window")
    if window is None:
        left = int(op.kwargs.get("window_left", -1))
        right = int(op.kwargs.get("window_right", -1))
    elif isinstance(window, (tuple, list)):
        left, right = map(int, window)
    else:
        left = right = int(window)
    causal = bool(op.kwargs.get("causal", False))
    if target == "x86" and left != right:
        raise ValueError("Zen 5 attention backward requires a symmetric window")
    if target == "rocm_gfx1151" and not (
        (left == -1 and right == -1) or (causal and left >= 0 and right == 0)
    ):
        raise ValueError("gfx1151 attention backward window is unsupported")
    scale = float(op.kwargs.get("scale", d ** -0.5))
    softcap = float(op.kwargs.get("softcap", op.kwargs.get("logit_softcap", 0.0)) or 0.0)
    dropout = float(op.kwargs.get("dropout_p", op.kwargs.get("dropout", 0.0)) or 0.0)
    seed = int(op.kwargs.get("dropout_seed", op.kwargs.get("seed", 0)) or 0)
    if not math.isfinite(scale) or scale <= 0 or not math.isfinite(softcap) or softcap < 0 or not 0 <= dropout < 1:
        raise ValueError("attention backward numeric modifiers are invalid")
    if target == "x86" and dropout:
        raise ValueError("Zen 5 attention backward does not support dropout")
    requested = op.kwargs.get("lse_checkpoint", "auto")
    if requested not in {"auto", "saved", "recompute"}:
        raise ValueError("invalid attention backward LSE checkpoint selection")
    if target == "x86":
        selection, policy = "saved", "save_lse"
        architecture, storage = "zen5-avx512", "f32"
    else:
        selection = "saved" if requested == "auto" and max(sq, sk) >= 128 else "recompute" if requested == "auto" else requested
        policy = "gfx1151_auto_128"
        architecture, storage = "gfx1151", {"fp16": "f16", "bf16": "bf16"}[dtype]
    return (
        "x86" if target == "x86" else "rocm", architecture, function.name,
        names, bias_name, result_names, dtype, storage,
        (b, hq, hkv, sq, sk, d, dv), scale, causal, left, right, softcap,
        dropout, seed, policy, selection,
    )


__all__ = [
    "ScheduledAttentionBackwardArtifact",
    "lower_scheduled_attention_backward",
    "supports_scheduled_attention_backward",
]
