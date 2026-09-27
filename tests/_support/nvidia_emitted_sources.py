"""Every CUDA source ``emit/nvidia_cuda.py`` emits, and its exported entries.

Shared by the host-independent rule gate
(``tests/unit/test_nvidia_emitted_stale_error_rule.py``), the sm_120 device
proof (``tests/device/nvidia/test_emitted_stale_cuda_error.py``) and the
evidence table under ``benchmarks/baselines/autotune_corpus_rerecord_sm120_*``
(sync ``SM120-AUTOTUNE-FOLLOWUPS-2026-09-27``).

Sources are rendered by calling every ``_synthesize_*`` emitter with
representative arguments. The table below must name every emitter in the
module: a new one that is not listed fails the gate, so the stale-error rule
cannot be skipped by adding an emitter. Two entries are built inline inside
run functions rather than by an emitter; they are checked from their Python
source text instead (``INLINE_SOURCE_FUNCTIONS``).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable

#: The one statement the rule allows as a clear.
CLEAR = "(void)cudaGetLastError();"

_READ = re.compile(r"cudaGetLastError\s*\(\s*\)|cudaPeekAtLastError\s*\(")
_EXTERN = re.compile(r'extern\s+"C"\s+(?!__global__)')
_LAUNCH = re.compile(r"<<<")
_RUNTIME_CALL = re.compile(r"\bcuda[A-Z]\w*\s*\(")
_CALL = re.compile(r"\b(\w+)\s*(?:<[^<>;]*>)?\s*\(")
_NOT_CALLS = frozenset({"if", "for", "while", "switch", "return", "sizeof"})

#: Run functions that hold an inline CUDA source (not an emitter).
INLINE_SOURCE_FUNCTIONS = ("run_fused_epilogue_f32", "run_relu_bias_f32")


def _emitters() -> dict[str, list[tuple[str, Callable[[], str]]]]:
    from tessera.compiler.emit import nvidia_cuda as N
    from tessera.compiler.fusion_core import (
        FusedRegion,
        GatedMatmulRegion,
        PointwiseGraphRegion,
    )

    pw = PointwiseGraphRegion(ops=(("add", ("x", "y"), "s"),
                                   ("relu", ("s",), "o")),
                              inputs=("x", "y"), output="o")
    lowp = ("f32", "fp8_e4m3", "fp8_e5m2")
    return {
        "_synthesize_fused_cuda": [("fused(bias,gelu)", lambda: N._synthesize_fused_cuda(
            FusedRegion(epilogue=("bias", "gelu"))))],
        "_synthesize_attention_cuda": [("attention", N._synthesize_attention_cuda)],
        "_synthesize_flash_fwd_cuda": [("flash_fwd", N._synthesize_flash_fwd_cuda)],
        "_synthesize_flash_fwd_f16_cuda": [("flash_fwd_f16", N._synthesize_flash_fwd_f16_cuda)],
        "_synthesize_flash_fwd_multiwarp_cuda": [
            (f"flash_fwd_w{w}", lambda w=w: N._synthesize_flash_fwd_multiwarp_cuda(w))
            for w in (4, 8)],
        "_synthesize_mla_fused_cuda": [("mla_fused", N._synthesize_mla_fused_cuda)],
        "_synthesize_flash_bwd_cuda": [("flash_bwd", N._synthesize_flash_bwd_cuda)],
        "_synthesize_flash_bwd_f16_cuda": [("flash_bwd_f16", N._synthesize_flash_bwd_f16_cuda)],
        "_synthesize_linear_attn_cuda": [("linear_attn", N._synthesize_linear_attn_cuda)],
        "_synthesize_linear_attn_bwd_cuda": [("linear_attn_bwd", N._synthesize_linear_attn_bwd_cuda)],
        "_synthesize_linear_attn_variant_cuda": [
            ("linear_attn_variant", N._synthesize_linear_attn_variant_cuda)],
        "_synthesize_linear_attn_variant_bwd_cuda": [
            ("linear_attn_variant_bwd", N._synthesize_linear_attn_variant_bwd_cuda)],
        "_synthesize_gated_cuda": [("gated(silu)", lambda: N._synthesize_gated_cuda(
            GatedMatmulRegion(gate_act="silu")))],
        "_synthesize_pointwise_cuda": [("pointwise(add,relu)",
                                        lambda: N._synthesize_pointwise_cuda(pw))],
        "_synthesize_softmax_cuda": [("softmax", N._synthesize_softmax_cuda)],
        "_synthesize_softmax_f16_cuda": [("softmax_f16", N._synthesize_softmax_f16_cuda)],
        "_synthesize_norm_cuda": [(f"norm_{d}", lambda d=d: N._synthesize_norm_cuda(d))
                                  for d in ("f32", "f16")],
        "_synthesize_reduce_cuda": [(f"reduce_{d}", lambda d=d: N._synthesize_reduce_cuda(d))
                                    for d in ("f32", "f16", "bf16")],
        "_synthesize_fpquant_cuda": [("fpquant", N._synthesize_fpquant_cuda)],
        "_synthesize_binary_cuda": [("binary", N._synthesize_binary_cuda)],
        "_synthesize_solver_ift_cuda": [("solver_ift", N._synthesize_solver_ift_cuda)],
        "_synthesize_solver_children_cuda": [
            ("solver_children", N._synthesize_solver_children_cuda)],
        "_synthesize_local_collective_cuda": [
            ("local_collective", N._synthesize_local_collective_cuda)],
        "_synthesize_optimizer_cuda": [("optimizer", N._synthesize_optimizer_cuda)],
        "_synthesize_dequant_grouped_cuda": [
            ("dequant_grouped", N._synthesize_dequant_grouped_cuda)],
        "_synthesize_moe_cuda": [("moe", N._synthesize_moe_cuda)],
        "_synthesize_deltanet_cuda": [("deltanet", N._synthesize_deltanet_cuda)],
        "_synthesize_ssm_cuda": [("ssm", N._synthesize_ssm_cuda)],
        "_synthesize_ssm_replay_decode_cuda": [
            ("ssm_replay_decode", N._synthesize_ssm_replay_decode_cuda)],
        "_synthesize_paged_kv_read_cuda": [("paged_kv_read", N._synthesize_paged_kv_read_cuda)],
        "_synthesize_gated_epilogue_cuda": [
            (f"gated_epilogue({a})", lambda a=a: N._synthesize_gated_epilogue_cuda(a)[2])
            for a in ("silu", "gelu")],
        "_synthesize_conv2d_nhwc_cuda": [("conv2d_nhwc", N._synthesize_conv2d_nhwc_cuda)],
        "_synthesize_ssm_replay_device_cuda": [
            ("ssm_replay_device", N._synthesize_ssm_replay_device_cuda)],
        "_synthesize_resident_ops_cuda": [("resident_ops", N._synthesize_resident_ops_cuda)],
        "_synthesize_posenc_cuda": [("posenc", N._synthesize_posenc_cuda)],
        "_synthesize_control_flow_cuda": [("control_flow", N._synthesize_control_flow_cuda)],
        "_synthesize_mma_fused_cuda": [
            (f"mma_fused({s},bias={b},{a})",
             lambda s=s, b=b, a=a: N._synthesize_mma_fused_cuda(b, a, s))
            for s in ("f16", "bf16", *lowp) for b, a in ((True, "gelu"), (False, None))],
        "_synthesize_mma_attn_16_cuda": [
            (f"mma_attn({s})", lambda s=s: N._synthesize_mma_attn_16_cuda(s))
            for s in ("f16", "bf16")],
        "_synthesize_mma_attn_lowp_cuda": [
            (f"mma_attn({s})", lambda s=s: N._synthesize_mma_attn_lowp_cuda(s)) for s in lowp],
        # Dispatches to the two above; rendered for completeness of the table.
        "_synthesize_mma_attn_cuda": [("mma_attn(dispatch f16)",
                                       lambda: N._synthesize_mma_attn_cuda("f16"))],
        "_synthesize_mma_gated_cuda": [
            (f"mma_gated({s},silu)", lambda s=s: N._synthesize_mma_gated_cuda(s, "silu"))
            for s in ("f16", "bf16", *lowp)],
    }


def emitter_names_in_module() -> set[str]:
    """Every ``_synthesize_*`` function defined in ``nvidia_cuda``."""
    from tessera.compiler.emit import nvidia_cuda as N

    return {name for name, value in vars(N).items()
            if name.startswith("_synthesize_") and callable(value)
            and getattr(value, "__module__", "") == N.__name__}


def rendered_sources() -> list[tuple[str, str, str]]:
    """``(emitter, label, source)`` for every representative emission."""
    out = []
    for emitter, calls in _emitters().items():
        for label, call in calls:
            out.append((emitter, label, call()))
    return out


def inline_source_text(function_name: str) -> str:
    import inspect

    from tessera.compiler.emit import nvidia_cuda as N

    return inspect.getsource(getattr(N, function_name))


@dataclass(frozen=True)
class Entry:
    name: str
    body: str               # between the outer braces

    @property
    def statements(self) -> str:
        return self.body.strip()

    def first_statement_is_clear(self) -> bool:
        return self.statements.startswith(CLEAR)

    def clears(self) -> int:
        return self.body.count(CLEAR)

    def reads(self) -> int:
        """Slot reads that are not the clear itself."""
        return len(_READ.findall(self.body)) - self.clears()

    def launches(self) -> int:
        return len(_LAUNCH.findall(self.body))

    def runtime_calls(self) -> int:
        return len(_RUNTIME_CALL.findall(self.body))

    def makes_calls(self) -> bool:
        """Whether the body calls anything at all -- a runtime function, a
        launch, or a helper that may make runtime calls."""
        return self.launches() > 0 or any(
            name not in _NOT_CALLS for name in _CALL.findall(self.body))


def _matching(text: str, open_at: int, open_ch: str, close_ch: str) -> int:
    depth = 0
    for i in range(open_at, len(text)):
        if text[i] == open_ch:
            depth += 1
        elif text[i] == close_ch:
            depth -= 1
            if depth == 0:
                return i
    raise ValueError("unbalanced source")


def exported_entries(source: str) -> list[Entry]:
    """Every ``extern "C"`` host function defined in ``source``."""
    entries = []
    for m in _EXTERN.finditer(source):
        paren = source.index("(", m.end())
        head = source[m.end():paren].split()
        name = head[-1].lstrip("*") if head else "?"
        close = _matching(source, paren, "(", ")")
        after = source[close + 1:].lstrip()
        if not after.startswith("{"):
            continue                     # a declaration, not a definition
        brace = source.index("{", close)
        end = _matching(source, brace, "{", "}")
        entries.append(Entry(name=name, body=source[brace + 1:end]))
    return entries


def source_reads_slot(source: str) -> bool:
    return len(_READ.findall(source)) > source.count(CLEAR)


def classify() -> list[dict[str, Any]]:
    """One row per exported entry of every rendered source (deduplicated by
    emitter and entry name, first representative wins)."""
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for emitter, label, source in rendered_sources():
        reads = source_reads_slot(source)
        for entry in exported_entries(source):
            if (emitter, entry.name) in seen:
                continue
            seen.add((emitter, entry.name))
            if not entry.makes_calls():
                kind = "metadata"
            elif reads:
                kind = "reads-slot"
            else:
                kind = "sync-only"
            rows.append(dict(emitter=emitter, label=label, entry=entry.name,
                             kind=kind, launches=entry.launches(),
                             reads=entry.reads(), clears=entry.clears(),
                             clear_first=entry.first_statement_is_clear()))
    return rows
