"""Fail-closed native ROCm dense operations for the one-step Gumiho demo.

The NumPy backend remains an independent numerical oracle, never an execution
fallback for this backend. Host arrays are currently the transport boundary.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np

from tessera import runtime as rt


class RocmBackend:
    def __init__(self, *, eps: float = 1e-5) -> None:
        self.eps = float(eps)
        self.arch = rt._rocm_live_arch()
        if not self.arch or not self.arch.startswith("gfx"):
            raise RuntimeError("Gumiho ROCm requires a visible AMD GPU")
        if rt._tessera_opt_path() is None:
            raise RuntimeError("Gumiho ROCm requires a built tessera-opt")
        configured = rt._rocm_chip().split(":", 1)[0]
        if configured != self.arch:
            raise RuntimeError(
                f"Gumiho ROCm compiler target {configured} differs from live {self.arch}"
            )
        self.name = f"rocm-{self.arch}"
        self.route_counts: Counter[str] = Counter()

    @staticmethod
    def _f32(value: Any) -> np.ndarray:
        return np.ascontiguousarray(value, dtype=np.float32)

    def _launch(self, path: str, op: str, values: tuple[np.ndarray, ...],
                *, kwargs: dict[str, Any] | None = None) -> np.ndarray:
        operands = ["x"] if len(values) == 1 else ["a", "b"]
        if op == "tessera.rmsnorm" and len(values) == 2:
            operands = ["x", "gamma"]
        artifact = rt.RuntimeArtifact(metadata={
            "target": "rocm", "compiler_path": path,
            "executable": True, "execution_kind": "native_gpu",
            "arg_names": operands, "output_name": "o",
            "ops": [{"op_name": op, "result": "o", "operands": operands,
                     "kwargs": kwargs or {}}],
        })
        result = rt.launch(artifact, values)
        if (not result.get("ok") or result.get("execution_kind") != "native_gpu"
                or result.get("compiler_path") != path):
            raise RuntimeError(f"Gumiho native {op} refused: {result.get('reason', result)}")
        self.route_counts[path] += 1
        return np.asarray(result["output"], dtype=np.float32).reshape(values[0].shape)

    def linear(self, x: Any, w: Any) -> np.ndarray:
        a, b = self._f32(x), self._f32(w)
        if a.ndim != 2 or b.ndim != 2:
            raise ValueError("Gumiho ROCm linear requires rank-2 operands")
        result = rt._rocm_f32_gemm(a, b, np)
        self.route_counts["rocm_f32_gemm"] += 1
        return result

    def matmul(self, a: Any, b: Any) -> np.ndarray:
        lhs, rhs = self._f32(a), self._f32(b)
        if lhs.ndim == rhs.ndim == 2:
            return self.linear(lhs, rhs)
        if lhs.ndim != 3 or rhs.ndim != 3:
            raise ValueError("Gumiho ROCm matmul requires matching rank-3 batches")
        result = rt._rocm_batched_gemm_f32(lhs, rhs, np)
        self.route_counts["rocm_batched_gemm_f32"] += 1
        return result

    def _binary(self, op: str, a: Any, b: Any) -> np.ndarray:
        lhs = self._f32(a)
        rhs = self._f32(np.broadcast_to(b, lhs.shape))
        return self._launch("rocm_binary_compiled", op, (lhs, rhs))

    def add(self, a: Any, b: Any) -> np.ndarray:
        return self._binary("tessera.add", a, b)

    def mul(self, a: Any, b: Any) -> np.ndarray:
        return self._binary("tessera.mul", a, b)

    def rmsnorm(self, x: Any, gamma: Any) -> np.ndarray:
        value = self._f32(x)
        scale = self._f32(gamma)
        return self._launch("rocm_norm_compiled", "tessera.rmsnorm",
                            (value, scale), kwargs={"eps": self.eps})

    def silu_mul(self, a: Any, b: Any) -> np.ndarray:
        return self._launch("rocm_silu_mul_compiled", "tessera.silu_mul",
                            (self._f32(a), self._f32(b)))

    def relu(self, x: Any) -> np.ndarray:
        return self._launch("rocm_activation_compiled", "tessera.relu",
                            (self._f32(x),))

    def softmax(self, x: Any) -> np.ndarray:
        return self._launch("rocm_softmax_compiled", "tessera.softmax",
                            (self._f32(x),), kwargs={"axis": -1})
