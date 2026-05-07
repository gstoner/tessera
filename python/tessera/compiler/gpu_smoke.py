"""Hardware-gated executable smoke paths for GPU compiler targets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .capabilities import CAPABILITY_REGISTRY_VERSION, normalize_target


@dataclass(frozen=True)
class SmokeResult:
    target: str
    op_name: str
    runtime_status: str
    ok: bool
    reason: str
    telemetry: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "op_name": self.op_name,
            "runtime_status": self.runtime_status,
            "ok": self.ok,
            "reason": self.reason,
            "telemetry": dict(self.telemetry),
        }


def run_matmul_smoke(target: object = "nvidia_sm90", *, size: int = 16) -> SmokeResult:
    """Run a minimal matmul smoke when the requested runtime is present.

    NVIDIA uses torch CUDA when available. Apple uses the Tessera apple_gpu JIT
    metadata path, which dispatches through the existing runtime shim on Darwin
    and falls back to a defined runtime result elsewhere.
    """

    target_name = normalize_target(target)
    if target_name.startswith("nvidia"):
        return _run_nvidia_torch_smoke(target_name, size=size)
    if target_name == "apple_gpu":
        return _run_apple_gpu_smoke(size=size)
    return _artifact_only(target_name, "unsupported smoke target")


def _run_nvidia_torch_smoke(target_name: str, *, size: int) -> SmokeResult:
    try:
        import torch
    except Exception as exc:
        return _artifact_only(target_name, f"torch is unavailable: {exc}")
    if not torch.cuda.is_available():
        return _artifact_only(target_name, "CUDA runtime is not available")
    a = torch.arange(size * size, dtype=torch.float32, device="cuda").reshape(size, size)
    b = torch.eye(size, dtype=torch.float32, device="cuda")
    out = a @ b
    expected = a.cpu().numpy()
    ok = np.allclose(out.cpu().numpy(), expected)
    return SmokeResult(
        target=target_name,
        op_name="tessera.matmul",
        runtime_status="ready" if ok else "invalid_artifact",
        ok=bool(ok),
        reason="torch CUDA matmul smoke executed",
        telemetry=_telemetry(target_name, "ready" if ok else "invalid_artifact"),
    )


def _run_apple_gpu_smoke(*, size: int) -> SmokeResult:
    import tessera as ts

    @ts.jit(target="apple_gpu")
    def mm(a, b):
        return ts.ops.matmul(a, b)

    a = np.arange(size * size, dtype=np.float32).reshape(size, size)
    b = np.eye(size, dtype=np.float32)
    try:
        out = mm(a, b)
    except Exception as exc:
        return _artifact_only("apple_gpu", f"Apple GPU runtime smoke did not execute: {exc}")
    ok = np.allclose(out, a)
    return SmokeResult(
        target="apple_gpu",
        op_name="tessera.matmul",
        runtime_status="ready" if ok else "invalid_artifact",
        ok=bool(ok),
        reason="Apple GPU matmul smoke executed through Tessera runtime path",
        telemetry=_telemetry("apple_gpu", "ready" if ok else "invalid_artifact"),
    )


def _artifact_only(target_name: str, reason: str) -> SmokeResult:
    return SmokeResult(
        target=target_name,
        op_name="tessera.matmul",
        runtime_status="artifact_only",
        ok=False,
        reason=reason,
        telemetry=_telemetry(target_name, "artifact_only"),
    )


def _telemetry(target_name: str, status: str) -> dict[str, Any]:
    return {
        "schema": "tessera.telemetry.v1",
        "name": "compiler.gpu_smoke",
        "source": "compiler",
        "op": "matmul",
        "arch": target_name,
        "status": status,
        "metadata": {"capability_version": CAPABILITY_REGISTRY_VERSION},
    }


__all__ = ["SmokeResult", "run_matmul_smoke"]
