"""TPP space-time arbiter candidates — the tpp-space-time retarget.

Points the TPP solver's ``lower-tpp-to-target-ir`` seam at the Workstream-C D1
arbiter (Decision #28): the stencil Target-IR call symbol the pass emits
(``ts_stencil_grad_cpu``) is registered here as an arbiter *candidate* for the
``tpp_stencil`` op-kind, F4-gated against a numpy central-difference reference.
The arbiter — not the pass — owns candidate selection per ``(op="tpp_stencil",
target)``, so a future WMMA/vectorised stencil lane slots in beside this one and
wins only when it measures faster and in budget.

The CPU lane runs the *real shipped kernel* (``TargetHooks/CPU/Stencil.cpp``)
via ctypes, so the Target-IR symbol the pass names has a verified
implementation.  Off a C++ toolchain it declines to the reference (host-free).
"""
from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from tessera.compiler.emit.candidate import (
    Candidate,
    Tier,
    register_candidate,
    register_op_kind,
    verify_by_reference,
)

#: op-kind tag emitted by LowerTPPToTargetIR (tessera.target_ir.arbiter_op).
OP_TPP_STENCIL = "tpp_stencil"

_REPO_ROOT = Path(__file__).resolve().parents[4]
_CPU_SRC = (_REPO_ROOT / "src" / "solvers" / "tpp" / "lib" / "TargetHooks"
            / "CPU" / "Stencil.cpp")


# --- region + reference ------------------------------------------------------
def central_difference_taps(
    accuracy_order: int,
    derivative_order: int,
    spacing: float,
) -> tuple[tuple[int, ...], tuple[float, ...]]:
    """Return physical central-difference taps for a one-dimensional axis.

    The returned coefficient contract is exactly the one consumed by
    ``tessera.neighbors.stencil.define``: coefficients include the dimensional
    factor ``spacing ** -derivative_order``.  Keeping derivative order distinct
    from accuracy order prevents the common error of dividing a Laplacian by
    ``h`` instead of ``h**2``.
    """
    h = float(spacing)
    if not np.isfinite(h) or h <= 0.0:
        raise ValueError("spacing must be finite and positive")
    key = (int(derivative_order), int(accuracy_order))
    numerators: dict[
        tuple[int, int], tuple[tuple[int, ...], tuple[float, ...], float]
    ] = {
        (1, 2): ((-1, 1), (-1.0, 1.0), 2.0),
        (1, 4): ((-2, -1, 1, 2), (1.0, -8.0, 8.0, -1.0), 12.0),
        (2, 2): ((-1, 0, 1), (1.0, -2.0, 1.0), 1.0),
        (2, 4): ((-2, -1, 0, 1, 2),
                 (-1.0, 16.0, -30.0, 16.0, -1.0), 12.0),
    }
    if key not in numerators:
        raise ValueError(
            "supported (derivative_order, accuracy_order) pairs are "
            "(1,2), (1,4), (2,2), and (2,4)"
        )
    offsets, weights, denominator = numerators[key]
    scale = denominator * (h ** derivative_order)
    return offsets, tuple(weight / scale for weight in weights)


class StencilGradRegion:
    """A periodic central-difference gradient of a 2-D field along ``axis`` with
    accuracy ``order`` (2 or 4), explicit per-axis grid spacing — the semantics of ``tpp.grad``
    after halo-infer + a local ``tpp.halo.exchange``."""

    def __init__(self, nx: int, ny: int, axis: int = 0, order: int = 2,
                 spacing: tuple[float, float] = (1.0, 1.0)):
        self.nx, self.ny, self.axis, self.order = int(nx), int(ny), int(axis), int(order)
        self.spacing = tuple(float(h) for h in spacing)
        if len(self.spacing) != 2 or any(not np.isfinite(h) or h <= 0.0
                                         for h in self.spacing):
            raise ValueError("spacing must contain two finite positive values")

    def reference(self, f: np.ndarray) -> np.ndarray:
        f = np.asarray(f, np.float32)
        offsets, coeffs = central_difference_taps(
            self.order, 1, self.spacing[self.axis]
        )
        g = np.zeros_like(f, dtype=np.float32)
        for offset, coeff in zip(offsets, coeffs, strict=True):
            g += np.float32(coeff) * np.roll(f, -offset, axis=self.axis)
        return g

    def probe_input(self, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return rng.standard_normal((self.nx, self.ny)).astype(np.float32)


def _verify_stencil(candidate: Candidate, region: StencilGradRegion, *,
                    atol: float = 1e-3, seed: int = 0) -> bool:
    f = region.probe_input(seed)
    ref = region.reference(f)
    return verify_by_reference(candidate, region, (f,), ref, atol=max(atol, 1e-4))


register_op_kind(OP_TPP_STENCIL, _verify_stencil)


# --- compile cache -----------------------------------------------------------
_lib: list[ctypes.CDLL | None] = []


def _cpu_lib() -> ctypes.CDLL | None:
    if _lib:
        return _lib[0]
    lib: ctypes.CDLL | None = None
    cxx = os.environ.get("CXX", "c++")
    if shutil.which(cxx) and _CPU_SRC.exists():
        try:
            d = tempfile.mkdtemp(prefix="tessera_tpp_stencil_")
            so = os.path.join(d, "libtpp_stencil.so")
            subprocess.check_call(
                [cxx, "-O2", "-std=c++17", "-shared", "-fPIC", str(_CPU_SRC),
                 "-o", so], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            lib = ctypes.CDLL(so)
            lib.ts_stencil_grad_cpu.restype = ctypes.c_int
            lib.ts_stencil_grad_cpu.argtypes = [
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
                ctypes.c_int, ctypes.c_int, ctypes.c_float]
        except Exception:
            lib = None
    _lib.append(lib)
    return lib


def _cptr(a: np.ndarray) -> ctypes.c_void_p:
    return a.ctypes.data_as(ctypes.c_void_p)


class TesseraStencilHookError(RuntimeError):
    """The CPU stencil hook rejected its arguments or the requested order."""


# --- candidate ---------------------------------------------------------------
class CpuStencilGradCandidate(Candidate):
    """Tier-1: the shipped CPU central-difference stencil kernel, compiled +
    dlopened.  Runs the real ``ts_stencil_grad_cpu`` the Target-IR lowering names."""

    name = "cpu_stencil_grad"
    tier = Tier.SYNTHESIZED
    target = "cpu"
    op = OP_TPP_STENCIL

    def available(self) -> bool:
        return _cpu_lib() is not None

    def run(self, region: StencilGradRegion, f: np.ndarray, *a: Any,
            **k: Any) -> tuple[Any, str]:
        lib = _cpu_lib()
        if lib is None:
            return region.reference(f), "reference"
        try:
            fin = np.ascontiguousarray(f, np.float32)
            out = np.empty((region.nx, region.ny), np.float32)
            status = lib.ts_stencil_grad_cpu(
                _cptr(fin), _cptr(out), region.nx, region.ny, region.axis,
                region.order, region.spacing[region.axis])
            if status != 0:
                # The hook refuses an order it does not implement rather than
                # returning a lower-order answer under the requested order's
                # name, so a non-zero status must not be smoothed over.
                raise TesseraStencilHookError(
                    f"ts_stencil_grad_cpu refused order={region.order} "
                    f"axis={region.axis} nx={region.nx} ny={region.ny} "
                    f"(status {status})"
                )
            return out, "cpu_stencil_grad"
        except TesseraStencilHookError:
            raise
        except Exception:
            return region.reference(f), "reference"


register_candidate(CpuStencilGradCandidate())
