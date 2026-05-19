"""Regression coverage for the 2026-05-19 static review findings.

Source-level guards for changes that don't have a Python-side
behavior to exercise directly (the runtime is a static archive
that we don't link from Python yet):

  * **P1 #1** — ``Backend::launchHostKernel`` returns ``TsrStatus``,
    CUDA/HIP return ``UNIMPLEMENTED``, and the C ABI propagates.
  * **P1 #2** — ``tsrShutdown`` refuses with ``INVALID_ARGUMENT``
    when ``g_live_streams + g_live_events + g_live_buffers > 0``.
  * **P1 #3** — ``Backend::consumeLastError`` exists and CUDA/HIP
    override it; ``tessera_runtime.cpp`` calls
    ``_PropagateBackendError`` after every previously-silent void
    backend call.
  * **P2 #4** — ``tessera_qos_limit_set`` clamps non-positive
    tokens to 1 before reaching ``TokenLimiter::set``.
  * **P2 #5** — ``ExecRuntime::submit`` snapshots ``nccl_`` /
    ``rccl_`` under ``mu_`` before calling into them.

Each test reads the relevant source file and asserts on the
contract surface; a future refactor that drops one of these
guards fails here before destabilizing the runtime.
"""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_CPP = REPO_ROOT / "src" / "runtime" / "src" / "tessera_runtime.cpp"
BASE_BACKEND_H = REPO_ROOT / "src" / "runtime" / "src" / "backend" / "base_backend.h"
CPU_CPP = REPO_ROOT / "src" / "runtime" / "src" / "backend" / "cpu_backend.cpp"
CUDA_CPP = REPO_ROOT / "src" / "runtime" / "src" / "backend" / "cuda_backend.cpp"
HIP_CPP = REPO_ROOT / "src" / "runtime" / "src" / "backend" / "hip_backend.cpp"
EXEC_CPP = (
    REPO_ROOT / "src" / "collectives" / "lib" / "Dialect"
    / "Collective" / "Runtime" / "Execution.cpp"
)
EXEC_H = (
    REPO_ROOT / "src" / "collectives" / "include" / "tessera"
    / "Dialect" / "Collective" / "Runtime" / "Execution.h"
)


# ---------------------------------------------------------------------------
# P1 #1: launchHostKernel returns TsrStatus
# ---------------------------------------------------------------------------

def test_launch_host_kernel_returns_status_in_base_interface() -> None:
    text = BASE_BACKEND_H.read_text(encoding="utf-8")
    assert "virtual TsrStatus launchHostKernel" in text, (
        "Backend::launchHostKernel must return TsrStatus (was void) so "
        "CUDA/HIP can refuse host tile kernels with UNIMPLEMENTED — "
        "see P1 #1 from the 2026-05-19 static review."
    )


def test_cuda_hip_refuse_host_tile_kernels() -> None:
    for path in (CUDA_CPP, HIP_CPP):
        text = path.read_text(encoding="utf-8")
        assert "return TSR_STATUS_UNIMPLEMENTED;" in text, (
            f"{path.name}: must return TSR_STATUS_UNIMPLEMENTED from "
            f"launchHostKernel; the previous code invoked the kernel "
            f"with the wrong ABI (fn(params, payload) instead of "
            f"fn(ctx, tile, thread))."
        )


def test_c_abi_propagates_launch_status() -> None:
    text = RUNTIME_CPP.read_text(encoding="utf-8")
    assert "TsrStatus st = s->dev->be->launchHostKernel" in text, (
        "tsrLaunchHostTileKernel must capture the backend's TsrStatus "
        "instead of unconditionally returning SUCCESS."
    )
    assert "if (st == TSR_STATUS_UNIMPLEMENTED)" in text


# ---------------------------------------------------------------------------
# P1 #2: tsrShutdown refuses on live handles
# ---------------------------------------------------------------------------

def test_shutdown_refuses_with_live_handles() -> None:
    text = RUNTIME_CPP.read_text(encoding="utf-8")
    assert "refusing to destroy devices with live handles" in text, (
        "tsrShutdown must emit a precise diagnostic naming live "
        "stream/event/buffer counts before tearing devices down — "
        "otherwise outstanding handles dereference freed memory."
    )
    # Counters must be incremented at create sites and decremented at
    # destroy sites; check the symbol surface.
    for sym in ("g_live_streams", "g_live_events", "g_live_buffers"):
        assert sym in text, f"missing live-handle counter {sym!r}"


def test_handle_counters_paired_at_create_and_destroy() -> None:
    """For each counter, there must be at least one ``++`` (create)
    and at least one ``--`` (destroy) site.  This catches the
    failure mode where someone adds a new ``tsrCreate*`` without
    its matching ``tsrDestroy*`` decrement."""
    text = RUNTIME_CPP.read_text(encoding="utf-8")
    for counter in ("g_live_streams", "g_live_events", "g_live_buffers"):
        assert f"++{counter}" in text, (
            f"no create-side ``++{counter}`` site; live-handle "
            f"counting will under-count and let tsrShutdown silently "
            f"destroy devices with outstanding handles."
        )
        assert f"--{counter}" in text, (
            f"no destroy-side ``--{counter}`` site; live-handle "
            f"counting will over-count and trap tsrShutdown forever."
        )


# ---------------------------------------------------------------------------
# P1 #3: consumeLastError surfaces CUDA/HIP errors
# ---------------------------------------------------------------------------

def test_consume_last_error_exists_in_base_interface() -> None:
    text = BASE_BACKEND_H.read_text(encoding="utf-8")
    assert "virtual TsrStatus consumeLastError" in text


def test_cuda_hip_override_consume_last_error() -> None:
    cuda_text = CUDA_CPP.read_text(encoding="utf-8")
    hip_text = HIP_CPP.read_text(encoding="utf-8")
    assert "TsrStatus consumeLastError" in cuda_text
    assert "TsrStatus consumeLastError" in hip_text
    assert "cudaGetErrorString" in cuda_text
    assert "hipGetErrorString" in hip_text


def test_c_abi_propagates_backend_errors_at_silent_sites() -> None:
    text = RUNTIME_CPP.read_text(encoding="utf-8")
    # The helper must exist and be called from every previously
    # silent void-returning entry point.
    assert "static TsrStatus _PropagateBackendError" in text
    for site in (
        "tsrFree",
        "tsrMemset",
        "tsrMemcpy",
        "tsrStreamSynchronize",
        "tsrRecordEvent",
        "tsrWaitEvent",
        "tsrEventSynchronize",
    ):
        # The function body must mention _PropagateBackendError after
        # the backend call — we don't try to AST-parse the C++, just
        # confirm the symbol appears in the entry point's vicinity.
        idx = text.find(f"TsrStatus {site}(")
        assert idx >= 0, f"could not find {site} in runtime.cpp"
        body_end = text.find("\n}\n", idx)
        body = text[idx:body_end]
        assert "_PropagateBackendError" in body, (
            f"{site}: still returns TSR_STATUS_SUCCESS unconditionally; "
            f"CUDA/HIP errors will be swallowed.  Route the return "
            f"through ``_PropagateBackendError(dev)``."
        )


# ---------------------------------------------------------------------------
# P2 #4: tessera_qos_limit_set clamps to >= 1
# ---------------------------------------------------------------------------

def test_qos_limit_set_clamps_nonpositive_tokens() -> None:
    text = EXEC_CPP.read_text(encoding="utf-8")
    # The clamp must appear before the call into the runtime.
    qos_idx = text.find("void tessera_qos_limit_set(int tokens)")
    assert qos_idx >= 0
    body = text[qos_idx:qos_idx + 800]
    assert "tokens < 1" in body or "tokens <= 0" in body, (
        "tessera_qos_limit_set must clamp non-positive tokens "
        "before they reach TokenLimiter::set — otherwise "
        "TokenLimiter::acquire deadlocks waiting for tokens_>0."
    )


# ---------------------------------------------------------------------------
# P2 #5: ExecRuntime::submit snapshots adapters under mu_
# ---------------------------------------------------------------------------

def test_submit_snapshots_adapters_under_lock() -> None:
    text = EXEC_H.read_text(encoding="utf-8")
    submit_idx = text.find("void submit(const ChunkDesc& d)")
    assert submit_idx >= 0
    body_end = text.find("  // Exposed to C hooks", submit_idx)
    body = text[submit_idx:body_end] if body_end > 0 else text[submit_idx:]
    # The snapshot block must take the internal mutex.
    assert "std::lock_guard<std::mutex>" in body
    assert "nccl_snap" in body and "rccl_snap" in body, (
        "ExecRuntime::submit must snapshot nccl_/rccl_ under mu_ "
        "before calling into them — otherwise setNCCL/setRCCL "
        "writes race the read here."
    )
