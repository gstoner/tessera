"""NUMPOL-CARRIER-1 (integrated-plan queue row 3b) — the carried accumulator
changes what the machine computes, proven by EXECUTING it.

The fixtures in `tests/tessera-ir/phase2/numeric_policy_reduction_carrier.mlir`
pin the emitted types. Types are not the claim, though: the claim is that
honouring `numeric_policy.accum` fixes a numerical defect. So this row compiles
both lowerings the rest of the way — linalg → LLVM → native object — and runs
them.

The oracle can fail and is not of our choosing: **a softmax row sums to 1.**
Measured on the Strix Halo box (Zen 5, AVX-512):

    no policy  (accumulates in bf16 storage)   row sum = 1.466101
    accum="fp32" (the declared policy)         row sum = 1.000169

bf16 carries 8 significand bits, so the running sum of 4096 exponentials
stagnates once it exceeds ~256x the increment — the defining property of the
function is violated by 47%. The residual 1.7e-04 on the honoured lane is the
bf16 *storage* rounding of the output, which is what the policy actually asked
for and is not a defect.

Scope: this is a CPU correctness row on this host's toolchain. It says nothing
about any GPU backend. Skips cleanly — and loudly — when the toolchain needed
to produce a native object is absent; a skipped row is not a passed row.
"""

from __future__ import annotations

import ctypes
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

N = 4096
_TOOLS = ("mlir-opt", "mlir-translate", "llc", "clang")


def _tessera_opt() -> str:
    from tessera import runtime as rt

    path = rt._tessera_opt_path()
    if path is None:
        pytest.skip("tessera-opt not built")
    return str(path)


def _toolchain_or_skip() -> None:
    missing = [t for t in _TOOLS if shutil.which(t) is None]
    if missing:
        pytest.skip(f"native toolchain unavailable: {', '.join(missing)}")


_KERNELS = """
module {{
  func.func @{name}(%x: tensor<1x{n}xbf16>) -> tensor<1x{n}xf32> {{
    %s = "tessera.softmax"(%x) {{axis = 1 : i64{policy}}}
      : (tensor<1x{n}xbf16>) -> tensor<1x{n}xbf16>
    %e = tensor.empty() : tensor<1x{n}xf32>
    %o = linalg.generic {{indexing_maps = [affine_map<(d0,d1)->(d0,d1)>,
                                           affine_map<(d0,d1)->(d0,d1)>],
                          iterator_types = ["parallel","parallel"]}}
         ins(%s : tensor<1x{n}xbf16>) outs(%e : tensor<1x{n}xf32>) {{
      ^bb0(%a: bf16, %b: f32):
        %c = arith.extf %a : bf16 to f32
        linalg.yield %c : f32
    }} -> tensor<1x{n}xf32>
    return %o : tensor<1x{n}xf32>
  }}
}}
"""

_BUFFERIZE = [
    '-one-shot-bufferize=bufferize-function-boundaries',
    "-convert-linalg-to-loops", "-convert-scf-to-cf",
    "-expand-strided-metadata", "-lower-affine",
    "-finalize-memref-to-llvm", "-convert-math-to-llvm",
    "-convert-arith-to-llvm", "-convert-cf-to-llvm",
    "-convert-func-to-llvm", "-reconcile-unrealized-casts",
]


class _Memref2D(ctypes.Structure):
    _fields_ = [
        ("alloc", ctypes.c_void_p), ("align", ctypes.c_void_p),
        ("offset", ctypes.c_longlong),
        ("sizes", ctypes.c_longlong * 2), ("strides", ctypes.c_longlong * 2),
    ]


def _build(tmp: Path, name: str, policy: str) -> ctypes.CDLL:
    src = tmp / f"{name}.mlir"
    src.write_text(_KERNELS.format(name=name, n=N, policy=policy))
    run = lambda cmd: subprocess.run(cmd, check=True, capture_output=True)
    run([_tessera_opt(), str(src), "--tessera-to-linalg",
         "-o", str(tmp / f"{name}.linalg.mlir")])
    run(["mlir-opt", str(tmp / f"{name}.linalg.mlir"), *_BUFFERIZE,
         "-o", str(tmp / f"{name}.llvm.mlir")])
    run(["mlir-translate", "--mlir-to-llvmir", str(tmp / f"{name}.llvm.mlir"),
         "-o", str(tmp / f"{name}.ll")])
    run(["llc", "-filetype=obj", "-O2", str(tmp / f"{name}.ll"),
         "-o", str(tmp / f"{name}.o")])
    run(["clang", "-shared", "-fPIC", str(tmp / f"{name}.o"),
         "-o", str(tmp / f"lib{name}.so"), "-lm"])
    return ctypes.CDLL(str(tmp / f"lib{name}.so"))


def _run(lib: ctypes.CDLL, name: str, x: np.ndarray) -> np.ndarray:
    fn = getattr(lib, name)
    fn.restype = None
    fn.argtypes = [ctypes.POINTER(_Memref2D), ctypes.c_void_p, ctypes.c_void_p,
                   *([ctypes.c_longlong] * 5)]
    out = _Memref2D()
    ptr = x.ctypes.data_as(ctypes.c_void_p)
    fn(ctypes.byref(out), ptr, ptr, 0, 1, N, N, 1)
    buf = (ctypes.c_float * N).from_address(out.align)
    return np.frombuffer(bytearray(buf), dtype=np.float32).copy()


@pytest.fixture(scope="module")
def _lanes(tmp_path_factory):
    _toolchain_or_skip()
    tmp = tmp_path_factory.mktemp("numpol")
    return (
        _build(tmp, "sm_storage_accum", ""),
        _build(tmp, "sm_declared_accum",
               ', numeric_policy = {storage = "bf16", accum = "fp32"}'),
    )


def _input():
    ml_dtypes = pytest.importorskip("ml_dtypes")
    rs = np.random.RandomState(7)
    return (rs.randn(1, N) * 2.0).astype(ml_dtypes.bfloat16)


def test_accumulating_in_storage_violates_the_softmax_identity(_lanes):
    """The control that must FAIL the oracle. Without it, the row below would
    pass on a compiler that had simply never had the defect, and would prove
    nothing about the carrier."""
    storage_lane, _ = _lanes
    y = _run(storage_lane, "sm_storage_accum", _input())
    row_sum = float(np.float64(y).sum())
    assert row_sum > 1.2, (
        f"row sum {row_sum} — the bf16-accumulator defect this carrier exists "
        f"to fix is not reproducible here, so the row below is not evidence")


def test_the_declared_accumulator_restores_the_softmax_identity(_lanes):
    """The claim: honouring numeric_policy.accum is the difference between a
    47%-wrong reduction and a correct one, in code the machine actually ran."""
    _, declared_lane = _lanes
    y = _run(declared_lane, "sm_declared_accum", _input())
    row_sum = float(np.float64(y).sum())
    # The tolerance is bf16 STORAGE rounding of the output, not slack: each of
    # the N outputs is rounded to 8 significand bits, so the sum carries
    # ~2^-9 of relative error. It is two orders below the defect above.
    assert abs(row_sum - 1.0) < 1e-3, f"row sum {row_sum}"


def test_the_carried_policy_beats_the_storage_accumulator_pointwise(_lanes):
    """Not just the aggregate identity: every element is closer to the fp64
    reference, so the win is not one lucky cancellation in the sum."""
    storage_lane, declared_lane = _lanes
    x = _input()
    exact = np.exp(np.float64(x) - np.float64(x).max())
    exact = (exact / exact.sum())[0]

    def rel(y):
        return np.abs(np.float64(y) - exact) / (exact + 1e-300)

    stored = rel(_run(storage_lane, "sm_storage_accum", x))
    declared = rel(_run(declared_lane, "sm_declared_accum", x))
    assert np.median(declared) < np.median(stored) / 100.0
    # ...and it is not merely better on average: it is better almost everywhere.
    assert float((declared < stored).mean()) > 0.95
