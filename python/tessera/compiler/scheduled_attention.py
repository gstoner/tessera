"""Canonical rank-4 attention through Graph -> Schedule -> launch Tile."""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass

from .graph_ir import GraphIRModule
from .scheduled_matmul import digest_text, find_tessera_opt, run_tessera_opt


_HASH_RE = re.compile(r'tessera\.schedule_hash = "([0-9a-f]{64})"')
_RECURRENCE = "rank4_batch_query_head_kv_online_softmax_v1"


@dataclass(frozen=True)
class ScheduledAttentionArtifact:
    graph_ir: str
    schedule_ir: str
    semantic_ir: str
    tile_ir: str
    target: str
    architecture: str
    function_name: str
    q_name: str
    k_name: str
    v_name: str
    bias_name: str | None
    output_name: str
    dtype: str
    storage: str
    accum: str
    dims: tuple[int, int, int, int, int, int, int]
    scale: float
    causal: bool
    window_left: int
    window_right: int
    softcap: float
    dropout_p: float
    dropout_seed: int
    tile_q: int
    tile_kv: int
    workgroup_size: int
    recurrence: str
    backward_lse_policy: str
    backward_lse_selection: str
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
        if len(re.findall(r"(?m)^\s*%[^=]+ = schedule\.attention\b", self.schedule_ir)) != 1:
            raise ValueError("scheduled attention requires one schedule.attention SSA edge")
        if len(re.findall(r"(?m)^\s*schedule\.artifact\b", self.schedule_ir)) != 1:
            raise ValueError("scheduled attention requires one durable schedule artifact")
        if self.schedule_ir.count(self.schedule_digest) != 3:
            raise ValueError("scheduled attention has incomplete content identity")
        if self.tile_ir.count("tile.attention_kernel") != 1:
            raise ValueError("scheduled attention requires one launch-level Tile op")
        if "tessera.flash_attn" in self.tile_ir or "schedule." in self.tile_ir:
            raise ValueError("scheduled attention Tile artifact retains Graph or Schedule IR")
        if _HASH_RE.findall(self.tile_ir) != [self.schedule_digest]:
            raise ValueError("scheduled attention Tile artifact has a stale schedule digest")
        required_recurrence = (
            'tessera.attention_distribution = "batch"',
            'tessera.attention_distribution = "query_head"',
            "tessera.streaming_attention",
            "tessera_attn.streaming_update",
            "scf.for",
        )
        if any(marker not in self.semantic_ir for marker in required_recurrence):
            raise ValueError("scheduled attention lost the shared rank-4 recurrence")
        modifier_markers = (
            (self.bias_name is not None, "tessera_attn.score_bias"),
            (self.softcap > 0.0, "tessera_attn.softcap"),
            (
                self.causal or self.window_left >= 0 or self.window_right >= 0,
                "tessera_attn.boundary_mask",
            ),
            (self.dropout_p > 0.0, "tessera_attn.block_dropout"),
        )
        if any(enabled and marker not in self.semantic_ir for enabled, marker in modifier_markers):
            raise ValueError("scheduled attention semantic recurrence dropped a modifier")
        for name, value in (
            ("tessera.tile_q", self.tile_q),
            ("tessera.tile_kv", self.tile_kv),
            ("tessera.workgroup_size", self.workgroup_size),
            ("head_dim", self.dims[5]),
            ("value_dim", self.dims[6]),
        ):
            if not re.search(rf"{re.escape(name)} = {value} : i64", self.tile_ir):
                raise ValueError(f"scheduled attention has stale {name}")
        for name, value in (
            ("tessera.attention_recurrence", self.recurrence),
            ("tessera.backward_lse_policy", self.backward_lse_policy),
            ("tessera.backward_lse_selection", self.backward_lse_selection),
        ):
            if f'{name} = "{value}"' not in self.tile_ir:
                raise ValueError(f"scheduled attention has stale {name}")
        if self.schedule_ir_digest == self.tile_digest:
            raise ValueError("Schedule and Tile attention artifacts must be distinct")


def supports_scheduled_attention(module: GraphIRModule, *, target: str) -> bool:
    try:
        _graph_contract(module, target)
    except ValueError:
        return False
    return True


def lower_scheduled_attention(
    module: GraphIRModule,
    *,
    target: str,
) -> ScheduledAttentionArtifact:
    contract = _graph_contract(module, target)
    tool = find_tessera_opt()
    if tool is None:
        raise RuntimeError("scheduled attention lowering requires production tessera-opt")

    targeted = copy.deepcopy(module)
    op = targeted.functions[0].body[0]
    op.kwargs = dict(op.kwargs)
    for alias in ("window", "logit_softcap", "dropout", "seed"):
        op.kwargs.pop(alias, None)
    op.kwargs.update(
        {
            "head_dim": contract[8][5],
            "scale": contract[11],
            "causal": contract[12],
            "window_left": contract[13],
            "window_right": contract[14],
            "softcap": contract[15],
            "dropout_p": contract[16],
            "dropout_seed": contract[17],
        }
    )
    targeted.module_attrs["tessera.target"] = f'"{contract[0]}"'
    targeted.module_attrs["tessera.arch"] = f'"{contract[1]}"'
    graph_ir = targeted.to_mlir(target=target, canonical=True)
    schedule_ir = run_tessera_opt(tool, graph_ir, "--tessera-graph-to-schedule")
    tile_ir = run_tessera_opt(tool, schedule_ir, "--tessera-schedule-to-tile")
    semantic_ir = run_tessera_opt(
        tool,
        graph_ir,
        (f"--tessera-tile-ir-lowering=tile-q={contract[18]} tile-kv={contract[19]} sm=90"),
    )
    hashes = _HASH_RE.findall(tile_ir)
    if len(hashes) != 1:
        raise RuntimeError("scheduled attention lowering lost its schedule identity")
    artifact = ScheduledAttentionArtifact(
        graph_ir=graph_ir,
        schedule_ir=schedule_ir,
        semantic_ir=semantic_ir,
        tile_ir=tile_ir,
        target=contract[0],
        architecture=contract[1],
        function_name=contract[2],
        q_name=contract[3][0],
        k_name=contract[3][1],
        v_name=contract[3][2],
        bias_name=contract[4],
        output_name=contract[5],
        dtype=contract[6],
        storage=contract[7],
        accum="f32",
        dims=contract[8],
        scale=contract[11],
        causal=contract[12],
        window_left=contract[13],
        window_right=contract[14],
        softcap=contract[15],
        dropout_p=contract[16],
        dropout_seed=contract[17],
        tile_q=contract[18],
        tile_kv=contract[19],
        workgroup_size=contract[20],
        recurrence=_RECURRENCE,
        backward_lse_policy=contract[21],
        backward_lse_selection=contract[22],
        schedule_digest=hashes[0],
    )
    artifact.validate()
    return artifact


def _graph_contract(module: GraphIRModule, target: str) -> tuple:
    if target == "x86":
        from .x86_native import _attention_contract

        physical = _attention_contract(module)
        if physical is None:
            raise ValueError("unsupported Zen 5 scheduled attention contract")
        names, bias_name, output_name, dims, scale, causal, window, softcap = physical
        dtype, storage = "fp32", "f32"
        window_left = window_right = window
        dropout_p, dropout_seed = 0.0, 0
        compiler_target, architecture = "x86", "zen5-avx512"
        workgroup_size = 1
        backward_lse_policy, backward_lse_selection = "save_lse", "saved"
    elif target == "rocm_gfx1151":
        from .rocm_native import _attention_contract

        physical = _attention_contract(module)
        if physical is None:
            raise ValueError("unsupported gfx1151 scheduled attention contract")
        (
            names,
            bias_name,
            dtype,
            dims,
            scale,
            causal,
            window_left,
            window_right,
            softcap,
            dropout_p,
            dropout_seed,
        ) = physical
        output_name = module.functions[0].body[0].result or "output"
        storage = {"fp16": "f16", "bf16": "bf16"}[dtype]
        compiler_target, architecture = "rocm", "gfx1151"
        workgroup_size = 256
        backward_lse_policy = "gfx1151_auto_128"
        backward_lse_selection = "saved" if dims[3] >= 128 else "recompute"
    else:
        raise ValueError(
            "scheduled attention supports only x86 and rocm_gfx1151; "
            "gfx1200/gfx1250 require architecture-owned profiles and evidence"
        )
    function = module.functions[0]
    tile_q, tile_kv = dims[3], 16
    return (
        compiler_target,
        architecture,
        function.name,
        names,
        bias_name,
        output_name,
        dtype,
        storage,
        dims,
        module.functions[0].result_types[0].dtype,
        _RECURRENCE,
        float(scale),
        bool(causal),
        int(window_left),
        int(window_right),
        float(softcap),
        float(dropout_p),
        int(dropout_seed),
        tile_q,
        tile_kv,
        workgroup_size,
        backward_lse_policy,
        backward_lse_selection,
    )
