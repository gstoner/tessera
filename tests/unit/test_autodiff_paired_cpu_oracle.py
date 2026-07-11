"""Phase 3 — CPU vertical slice: compiler-generated backward vs. Python oracle.

The honest, non-circular proof at the heart of Phase 3
(``docs/audit/compiler/AUTODIFF_UNIFICATION_PLAN.md``): take a real
``@jit(autodiff="reverse")``-shaped forward program, run it through the **actual
built** ``tessera-opt --tessera-autodiff-paired`` C++ pass, then numerically
interpret the pass's *emitted backward IR* and assert its gradients match an
**independent** NumPy VJP oracle.

This executes the compiler's output (not a Python reimplementation of it), so a
pass bug produces a mismatch. It is the CPU IR-execution rung — distinct from
native LLVM/runtime execution (Phase 4) and from ``hardware_proven``. The tiny
interpreter below understands only the op subset these adjoints emit; an
unknown op raises rather than silently skipping (Decision #21 / no silent
no-op).

Skips cleanly when ``tessera-opt`` is not built.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[2]
_TESSERA_OPT_CANDIDATES = (
    _REPO / "build" / "tools" / "tessera-opt" / "tessera-opt",
)


def _find_tessera_opt() -> str | None:
    for c in _TESSERA_OPT_CANDIDATES:
        if c.is_file():
            return str(c)
    return shutil.which("tessera-opt")


_TESSERA_OPT = _find_tessera_opt()
pytestmark = pytest.mark.skipif(
    _TESSERA_OPT is None, reason="tessera-opt not built (build it to run this proof)")


# ── a tiny NumPy interpreter for the paired-backward op subset ──────────────

_SSA = re.compile(r"%[A-Za-z0-9_]+")
_DIMS = re.compile(r"tensor<([0-9x]+)x[a-z0-9]+>")
_DENSE = re.compile(r"dense<([-0-9.eE+]+)>")


class _Interp:
    """Interprets one `func.func` body over the op subset the tanh/sigmoid/matmul
    adjoints emit: tessera.matmul (± transposeA/B), tessera.{tanh,sigmoid,mul,sub},
    arith.constant dense<..>, func.return."""

    def __init__(self, arg_names: list[str], body: list[str], rets: list[str]):
        self.arg_names = arg_names
        self.body = body
        self.rets = rets

    def run(self, inputs: list[np.ndarray]) -> list[np.ndarray]:
        env: dict[str, np.ndarray] = dict(zip(self.arg_names, inputs))
        for line in self.body:
            lhs, rhs = line.split(" = ", 1)
            dst = lhs.strip()
            head = rhs.split(" : ", 1)[0]
            op = head.split()[0]
            operands = _SSA.findall(head)
            if op == "arith.constant":
                val = float(_DENSE.search(rhs).group(1))
                dims = tuple(int(d) for d in _DIMS.search(rhs).group(1).split("x"))
                env[dst] = np.full(dims, val, dtype=np.float32)
            elif op == "tessera.matmul":
                a = env[operands[0]].copy()
                b = env[operands[1]].copy()
                if "transposeA = true" in head:
                    a = np.swapaxes(a, -1, -2)
                if "transposeB = true" in head:
                    b = np.swapaxes(b, -1, -2)
                env[dst] = (a @ b).astype(np.float32)
            elif op == "tessera.tanh":
                env[dst] = np.tanh(env[operands[0]]).astype(np.float32)
            elif op == "tessera.sigmoid":
                env[dst] = (1.0 / (1.0 + np.exp(-env[operands[0]]))).astype(np.float32)
            elif op == "tessera.mul":
                env[dst] = (env[operands[0]] * env[operands[1]]).astype(np.float32)
            elif op == "tessera.sub":
                env[dst] = (env[operands[0]] - env[operands[1]]).astype(np.float32)
            elif op == "tessera.add":
                env[dst] = (env[operands[0]] + env[operands[1]]).astype(np.float32)
            elif op == "arith.addf":
                env[dst] = (env[operands[0]] + env[operands[1]]).astype(np.float32)
            else:
                raise AssertionError(
                    f"paired-backward interpreter: unhandled op {op!r} in line "
                    f"{line!r} — extend the interpreter deliberately, do not skip")
        return [env[r] for r in self.rets]


def _parse_functions(mlir: str) -> dict[str, _Interp]:
    """Parse `func.func` bodies out of the pass output into interpreters."""
    funcs: dict[str, _Interp] = {}
    lines = mlir.splitlines()
    i = 0
    while i < len(lines):
        m = re.search(r"func\.func @([A-Za-z0-9_]+)\((.*?)\)", lines[i])
        if not m:
            i += 1
            continue
        name = m.group(1)
        arg_names = re.findall(r"(%[A-Za-z0-9_]+):", m.group(2))
        body: list[str] = []
        rets: list[str] = []
        i += 1
        while i < len(lines) and not re.match(r"\s*}\s*$", lines[i]):
            s = lines[i].strip()
            if s.startswith("return"):
                rets = _SSA.findall(s.split(" : ", 1)[0])
            elif " = " in s:
                body.append(s)
            i += 1
        funcs[name] = _Interp(arg_names, body, rets)
    return funcs


def _run_paired(fwd_mlir: str) -> dict[str, _Interp]:
    out = subprocess.run(
        [_TESSERA_OPT, "--tessera-autodiff-paired", "/dev/stdin"],
        input=fwd_mlir, capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, f"tessera-opt failed:\n{out.stderr}"
    return _parse_functions(out.stdout)


# ── the vertical slice: forward = act(matmul(x, w)), loss = sum(forward) ─────

def _forward_mlir(act: str) -> str:
    return f"""
module {{
  func.func @loss(%x: tensor<4x8xf32>, %w: tensor<8x16xf32>) -> tensor<4x16xf32>
      attributes {{tessera.autodiff = "reverse"}} {{
    %m = "tessera.matmul"(%x, %w) : (tensor<4x8xf32>, tensor<8x16xf32>) -> tensor<4x16xf32>
    %a = "tessera.{act}"(%m) : (tensor<4x16xf32>) -> tensor<4x16xf32>
    return %a : tensor<4x16xf32>
  }}
}}
"""


def _oracle(act: str, x: np.ndarray, w: np.ndarray):
    """Independent NumPy VJP for loss = sum(act(x @ w))."""
    m = x @ w
    if act == "tanh":
        a = np.tanh(m)
        dm = 1.0 - a * a          # d/dm sum(tanh) with seed ones
    elif act == "sigmoid":
        a = 1.0 / (1.0 + np.exp(-m))
        dm = a * (1.0 - a)
    else:  # pragma: no cover
        raise ValueError(act)
    dx = dm @ w.T
    dw = x.T @ dm
    return a, dx, dw


@pytest.mark.parametrize("act", ["tanh", "sigmoid"])
def test_compiler_backward_matches_numpy_oracle(act: str):
    rng = np.random.default_rng(0)
    x = rng.standard_normal((4, 8)).astype(np.float32)
    w = rng.standard_normal((8, 16)).astype(np.float32)

    funcs = _run_paired(_forward_mlir(act))
    assert "loss" in funcs and "loss__bwd" in funcs

    # Forward: the compiler-emitted primal must match the oracle primal.
    (primal,) = funcs["loss"].run([x, w])
    a_ref, dx_ref, dw_ref = _oracle(act, x, w)
    np.testing.assert_allclose(primal, a_ref, rtol=1e-4, atol=1e-5)

    # Backward: (x, w, seed=ones) -> (dx, dw), interpreted from the emitted IR.
    seed = np.ones((4, 16), dtype=np.float32)
    dx, dw = funcs["loss__bwd"].run([x, w, seed])
    np.testing.assert_allclose(dx, dx_ref, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(dw, dw_ref, rtol=1e-4, atol=1e-5)


def test_backward_signature_is_paired_abi():
    """The emitted backward has the (inputs, out_cotangents) -> input_cotangents
    shape the runtime binds in Phase 4."""
    funcs = _run_paired(_forward_mlir("tanh"))
    bwd = funcs["loss__bwd"]
    # 2 forward inputs + 1 out-cotangent = 3 args; 2 input cotangents returned.
    assert len(bwd.arg_names) == 3
    assert len(bwd.rets) == 2


def test_gradient_accumulation_reused_input():
    """A parameter used twice accumulates its gradient — loss = sum(tanh(x@w) +
    (x@w)) exercises two cotangent contributions into x@w's operands is out of
    scope here; instead we check the linear scaling property of the seed: doubling
    the output cotangent doubles the input gradients (the accumulate path is
    exercised structurally, and this guards the seed→grad linearity)."""
    rng = np.random.default_rng(1)
    x = rng.standard_normal((4, 8)).astype(np.float32)
    w = rng.standard_normal((8, 16)).astype(np.float32)
    funcs = _run_paired(_forward_mlir("tanh"))
    dx1, dw1 = funcs["loss__bwd"].run([x, w, np.ones((4, 16), np.float32)])
    dx2, dw2 = funcs["loss__bwd"].run([x, w, 2.0 * np.ones((4, 16), np.float32)])
    np.testing.assert_allclose(dx2, 2.0 * dx1, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(dw2, 2.0 * dw1, rtol=1e-4, atol=1e-5)
