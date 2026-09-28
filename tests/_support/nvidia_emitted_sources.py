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

#: Run functions that hold an inline CUDA source (not an emitter). Empty since
#: AUTOTUNE-LAUNCH-INTEGRITY-2026-09-27: the two that existed moved into
#: `_synthesize_relu_bias_cuda` / `_synthesize_fused_epilogue_cuda`, so every
#: rule in the gate reaches them.
INLINE_SOURCE_FUNCTIONS: tuple[str, ...] = ()


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
        "_synthesize_relu_bias_cuda": [("relu_bias", N._synthesize_relu_bias_cuda)],
        "_synthesize_fused_epilogue_cuda": [
            (f"fused_epilogue({a})", lambda a=a: N._synthesize_fused_epilogue_cuda(a)[1])
            for a in (None, "gelu")],
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


def functions_holding_kernel_source() -> list[str]:
    """Module functions other than ``_synthesize_*`` emitters whose own text
    holds CUDA kernel source (a ``__global__`` definition or a launch)."""
    import inspect

    from tessera.compiler.emit import nvidia_cuda as N

    out = []
    for name, value in vars(N).items():
        if (not callable(value) or name.startswith("_synthesize_")
                or getattr(value, "__module__", "") != N.__name__):
            continue
        try:
            text = inspect.getsource(value)
        except (OSError, TypeError):
            continue
        if "__global__" in text or "<<<" in text:
            out.append(name)
    return sorted(out)


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


# ── launch / status integrity (AUTOTUNE-LAUNCH-INTEGRITY-2026-09-27) ─────────
#
# Measured on sm_120 (CUDA 13.4 / driver 610.88): after a launch that never ran
# (invalid configuration), `cudaDeviceSynchronize` returns SUCCESS and the error
# sits only in the last-error slot. An entry that judges its launch by the sync
# alone reports a kernel that never executed as a success. So every runtime
# launch (`<<<...>>>`) is checked through the slot, and every call whose status
# says whether device memory holds what the kernel reads or writes -- an
# allocation, a copy, a memset, an event the timer depends on, a driver launch
# -- has its status consumed. A bare statement discards it.

#: Calls whose returned status must be consumed in a non-void host function.
STATUS_CALLS = (
    "cudaMalloc", "cudaMallocHost", "cudaHostAlloc", "cudaMallocAsync",
    "cudaMemcpy", "cudaMemcpyAsync", "cudaMemcpy2D", "cudaMemcpyToSymbol",
    "cudaMemset", "cudaMemsetAsync",
    "cudaEventCreate", "cudaEventCreateWithFlags", "cudaEventRecord",
    "cudaEventSynchronize", "cudaEventElapsedTime",
    "cudaStreamCreate", "cudaStreamSynchronize", "cudaDeviceSynchronize",
    "cuLaunchKernel",
)
_VOID_FUNCTION = re.compile(r'^\s*(?:extern\s+"C"\s+)?(?:static\s+)?(?:inline\s+)?void\s+\w+\s*\(')


@dataclass(frozen=True)
class Api:
    """The spelling of one runtime (CUDA or HIP) for the launch/status rules."""

    clear: str
    status: "re.Pattern[str]"
    read: "re.Pattern[str]"
    record: "re.Pattern[str]"
    reset: "re.Pattern[str]"
    launch: "re.Pattern[str]"


CUDA = Api(
    clear=CLEAR,
    status=re.compile(r"\b(" + "|".join(STATUS_CALLS) + r")\s*\("),
    read=re.compile(r"cudaGetLastError\s*\(\s*\)"),
    record=re.compile(r"\bcudaEventRecord\s*\("),
    reset=re.compile(r"\bcudaFuncSetAttribute\s*\("),
    launch=re.compile(r"<<<"),
)

#: The HIP spelling (`emit/rocm_hip.py`, sync `AUTOTUNE-LAUNCH-INTEGRITY-2026-09-27`).
#: HIP's slot is per host thread and sticky exactly like CUDA's.
HIP_STATUS_CALLS = tuple(
    c.replace("cuda", "hip", 1) for c in STATUS_CALLS if c.startswith("cuda")
) + ("hipHostMalloc", "hipStreamCreateWithFlags", "hipModuleLaunchKernel")
HIP = Api(
    clear="(void)hipGetLastError();",
    status=re.compile(r"\b(" + "|".join(HIP_STATUS_CALLS) + r")\s*\("),
    read=re.compile(r"hipGetLastError\s*\(\s*\)"),
    record=re.compile(r"\bhipEventRecord\s*\("),
    reset=re.compile(r"\bhipFuncSetAttribute\s*\("),
    launch=re.compile(r"<<<|\bhipLaunchKernelGGL\s*\("),
)


@dataclass(frozen=True)
class HostFunction:
    name: str
    signature: str          # text before the opening brace
    body: str               # between the outer braces

    @property
    def exported(self) -> bool:
        return 'extern "C"' in self.signature

    @property
    def returns_void(self) -> bool:
        return bool(_VOID_FUNCTION.match(self.signature))


def _strip_comments_and_literals(source: str) -> str:
    """``source`` with comments, string and char literals blanked to spaces,
    preserving every offset, so brace and call scanning cannot be misled by a
    ``{`` in a string or a call named in a comment."""
    out = list(source)
    i, n = 0, len(source)
    while i < n:
        c = source[i]
        if source.startswith("//", i):
            j = source.find("\n", i)
            j = n if j < 0 else j
            out[i:j] = " " * (j - i)
            i = j
        elif source.startswith("/*", i):
            j = source.find("*/", i + 2)
            j = n if j < 0 else j + 2
            out[i:j] = [ch if ch == "\n" else " " for ch in source[i:j]]
            i = j
        elif c in "\"'":
            j = i + 1
            while j < n and source[j] != c:
                j += 2 if source[j] == "\\" else 1
            out[i + 1:j] = " " * (j - i - 1)
            i = j + 1
        elif c == "#" and (i == 0 or source[i - 1] == "\n"):
            j = source.find("\n", i)       # preprocessor line: never code braces
            j = n if j < 0 else j
            out[i:j] = " " * (j - i)
            i = j
        else:
            i += 1
    return "".join(out)


def host_functions(source: str) -> list[HostFunction]:
    """Every top-level host function DEFINED in ``source`` (exported or
    static helper); ``__global__`` / ``__device__`` functions and struct
    bodies are skipped."""
    text = _strip_comments_and_literals(source)
    functions: list[HostFunction] = []
    boundary = 0
    i = 0
    while i < len(text):
        c = text[i]
        if c == ";":
            boundary = i + 1
        elif c == "{":
            end = _matching(text, i, "{", "}")
            head = text[boundary:i]
            stripped = head.rstrip()
            if stripped.endswith(")") and "(" in head:
                name_match = re.search(r"(\w+)\s*\($", head[:head.index("(")] + "(")
                name = name_match.group(1) if name_match else "?"
                if "__global__" not in head and "__device__" not in head:
                    functions.append(HostFunction(
                        name=name, signature=source[boundary:i],
                        body=source[i + 1:end]))
            i = end
            boundary = end + 1
        i += 1
    return functions


def _preceding_token(text: str, pos: int) -> str:
    j = pos - 1
    while j >= 0 and text[j].isspace():
        j -= 1
    if j < 0:
        return ""
    if text[j].isalnum() or text[j] == "_":
        k = j
        while k >= 0 and (text[k].isalnum() or text[k] == "_"):
            k -= 1
        return text[k + 1:j + 1]
    return text[j]


def unchecked_status_calls(function: HostFunction, api: Api = CUDA) -> list[str]:
    """Calls in :data:`STATUS_CALLS` whose status the function discards (a bare
    statement). Void functions (destructors) cannot report and are exempt."""
    if function.returns_void:
        return []
    body = _strip_comments_and_literals(function.body)
    bad = []
    for match in api.status.finditer(body):
        token = _preceding_token(body, match.start())
        if token in ("", ";", "{", "}", ":", ")", "else", "do"):
            bad.append(match.group(1))
    return bad


def launch_check_violations(function: HostFunction, api: Api = CUDA) -> list[str]:
    """How ``function``'s runtime launches escape a slot read, if they do.

    * the last launch must be followed by a slot read;
    * a launch group that ends at a timing boundary (an event record before
      the next launch) must be read before that boundary -- the warm-up group
      of a timer is checked before timing starts;
    * no ``cudaFuncSetAttribute`` may sit between a launch and its read (a
      successful one RESETS the slot, measured on sm_120)."""
    body = _strip_comments_and_literals(function.body)
    launches = [m.start() for m in api.launch.finditer(body)]
    if not launches:
        return []
    prefix = len("(void)")
    clears = {m.start() + prefix for m in re.finditer(re.escape(api.clear), body)}
    reads = [m.start() for m in api.read.finditer(body) if m.start() not in clears]
    records = [m.start() for m in api.record.finditer(body)]
    resets = [m.start() for m in api.reset.finditer(body)]
    problems = []
    if not any(r > launches[-1] for r in reads):
        problems.append("its last launch is never read through the slot")
    for here, after in zip(launches, launches[1:]):
        # A timing boundary BETWEEN two launch groups starts the timed region;
        # the group before it is read first. (After the last group, the event
        # record that ends timing may precede the read: a successful event
        # call leaves a launch error in the slot, measured on sm_120.)
        boundary = next((r for r in records if here < r < after), None)
        if boundary is not None and not any(here < r < boundary for r in reads):
            problems.append("a launch group reaches a timing boundary unread")
    for here in launches:
        nxt = next((r for r in reads if r > here), None)
        if nxt is not None and any(here < s < nxt for s in resets):
            problems.append("cudaFuncSetAttribute sits between a launch and its read")
    return sorted(set(problems))


def clears_before_first_launch(function: HostFunction, api: Api) -> bool:
    """Whether ``function`` discards the stale slot before its first launch.

    The HIP emitters clear after host-only argument validation, and the fused
    lane clears after its checked allocations and copies, immediately before
    launching. Either way no older error can be read back as the launch's, and
    every earlier call's status is consumed on its own
    (:func:`unchecked_status_calls`), so nothing the clear drops was unread."""
    body = _strip_comments_and_literals(function.body)
    launch = api.launch.search(body)
    clear = body.find(api.clear)
    return launch is None or (0 <= clear < launch.start())


def rocm_rendered_sources() -> list[tuple[str, str, str]]:
    """``(emitter, label, source)`` for every HIP source `emit/rocm_hip.py`
    renders; every ``_synthesize_*`` there must be listed."""
    from tessera.compiler.emit import rocm_hip as R
    from tessera.compiler.fusion_core import FusedRegion

    table: dict[str, list[tuple[str, Callable[[], str]]]] = {
        "_synthesize_fused_hip": [("fused(bias,gelu)", lambda: R._synthesize_fused_hip(
            FusedRegion(epilogue=("bias", "gelu"))))],
        "_synthesize_paged_kv_read_hip": [("paged_kv_read", R._synthesize_paged_kv_read_hip)],
        "_synthesize_paged_attention_direct_hip": [
            ("paged_attention_direct", R._synthesize_paged_attention_direct_hip)],
        "_synthesize_ssm_replay_device_hip": [
            ("ssm_replay_device", R._synthesize_ssm_replay_device_hip)],
    }
    defined = {name for name, value in vars(R).items()
               if name.startswith("_synthesize_") and callable(value)
               and getattr(value, "__module__", "") == R.__name__}
    missing = defined - set(table)
    if missing:
        raise AssertionError(
            f"HIP emitters not covered by the launch-integrity gate: {sorted(missing)}")
    return [(emitter, label, call()) for emitter, calls in table.items()
            for label, call in calls]


def classify() -> list[dict[str, Any]]:
    """One row per exported entry of every rendered source (deduplicated by
    emitter and entry name, first representative wins).

    Kinds: ``metadata`` (makes no call), ``slot-checked`` (in a source that
    reads the last-error slot: launches are checked through it) and
    ``status-checked`` (no runtime launch and no slot read: every call's own
    status is consumed -- the ReplaySSM ring, whose launches are driver
    ``cuLaunchKernel`` calls returning their own ``CUresult``). The
    ``sync-only`` class this gate used to accept -- a launch judged by
    ``cudaDeviceSynchronize`` alone -- no longer exists."""
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
                kind = "slot-checked"
            elif entry.launches() == 0 and "<<<" not in source:
                kind = "status-checked"
            else:
                kind = "sync-only"
            rows.append(dict(emitter=emitter, label=label, entry=entry.name,
                             kind=kind, launches=entry.launches(),
                             reads=entry.reads(), clears=entry.clears(),
                             clear_first=entry.first_statement_is_clear()))
    return rows
