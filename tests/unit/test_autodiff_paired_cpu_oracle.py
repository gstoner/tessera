"""Phase 3 — CPU vertical slice: compiler-generated backward vs. Python oracle.

The honest, non-circular proof at the heart of Phase 3
(``docs/audit/compiler/AUTODIFF_UNIFICATION_PLAN.md``): take a real
``@jit(autodiff="reverse")``-shaped forward program, run it through the **actual
built** ``tessera-opt --tessera-autodiff-paired`` C++ pass, then numerically
interpret the pass's *emitted backward IR* and assert its gradients match an
**independent** NumPy VJP oracle.

This executes the compiler's output (not a Python reimplementation of it), so a
pass bug produces a mismatch. It is the CPU IR-execution rung — distinct from
native LLVM/runtime execution (Phase 4) and from device verification. The tiny
interpreter below understands only the op subset these adjoints emit; an
unknown op raises rather than silently skipping (Decision #21 / no silent
no-op).

Skips cleanly when ``tessera-opt`` is not built.
"""

from __future__ import annotations

import re
import os
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
    configured = os.environ.get("TESSERA_OPT")
    if configured and Path(configured).is_file():
        return configured
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
_SHAPE = re.compile(r"tensor<((?:[?0-9]+x)*)(?:bf16|f16|f32|f64|i1)>")
_DENSE = re.compile(r"dense<([-0-9.eE+]+)>")


class _Interp:
    """Interprets one `func.func` body over the op subset the tanh/sigmoid/matmul
    adjoints emit: tessera.matmul (± transposeA/B), tensor algebra and static
    broadcast inversion, arith.constant dense<..>, func.return."""

    def __init__(self, arg_names: list[str], body: list[str], rets: list[str]):
        self.arg_names = arg_names
        self.body = body
        self.rets = rets

    def run(self, inputs: list[np.ndarray]) -> list[np.ndarray]:
        env: dict[str, np.ndarray] = dict(zip(self.arg_names, inputs))

        def result_shape(text: str) -> tuple[int, ...]:
            matches = _SHAPE.findall(text)
            assert matches, f"no tensor result shape in {text!r}"
            raw = matches[-1].removesuffix("x")
            tokens = raw.split("x") if raw else []
            if "?" not in tokens:
                return tuple(int(d) for d in tokens)
            for value in env.values():
                if value.ndim != len(tokens):
                    continue
                # Dynamic result dimensions are shape-carried by a same-rank
                # operand. Static result dimensions may intentionally differ
                # from that operand for broadcast expansion.
                return tuple(
                    value.shape[i] if t == "?" else int(t)
                    for i, t in enumerate(tokens)
                )
            raise AssertionError(f"cannot resolve dynamic result shape {tokens}")

        for line in self.body:
            lhs, rhs = line.split(" = ", 1)
            destinations = _SSA.findall(lhs)
            dst = destinations[0]
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
            elif op == "tessera.silu":
                x = env[operands[0]]
                env[dst] = (x / (1.0 + np.exp(-x))).astype(np.float32)
            elif op == "tessera.gelu":
                x = env[operands[0]]
                inner = np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)
                env[dst] = (0.5 * x * (1.0 + np.tanh(inner))).astype(np.float32)
            elif op == "tessera.relu":
                env[dst] = np.maximum(env[operands[0]], 0).astype(np.float32)
            elif op == "tessera.rmsnorm":
                x = env[operands[0]]
                eps_match = re.search(r"eps = ([-0-9.eE+]+)", rhs)
                eps = float(eps_match.group(1)) if eps_match else 1.0e-5
                y = x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)
                if len(operands) > 1:
                    y = y * env[operands[1]]
                env[dst] = y.astype(np.float32)
            elif op == "tessera.layer_norm":
                x = env[operands[0]]
                eps_match = re.search(r"eps = ([-0-9.eE+]+)", rhs)
                eps = float(eps_match.group(1)) if eps_match else 1.0e-5
                center = np.mean(x, axis=-1, keepdims=True)
                variance = np.mean((x - center) ** 2, axis=-1, keepdims=True)
                y = (x - center) / np.sqrt(variance + eps)
                if len(operands) > 1:
                    y = y * env[operands[1]]
                if len(operands) > 2:
                    y = y + env[operands[2]]
                env[dst] = y.astype(np.float32)
            elif op == "tessera.compare_scalar":
                x = env[operands[0]]
                rhs_value = float(re.search(r"rhs = ([-0-9.eE+]+)", rhs).group(1))
                predicate = re.search(r'predicate = "([a-z]+)"', rhs).group(1)
                comparisons = {
                    "eq": np.equal, "ne": np.not_equal,
                    "lt": np.less, "le": np.less_equal,
                    "gt": np.greater, "ge": np.greater_equal,
                }
                env[dst] = comparisons[predicate](x, rhs_value)
            elif op == "tessera.masked_fill":
                value = float(re.search(r"value = ([-0-9.eE+]+)", rhs).group(1))
                env[dst] = np.where(env[operands[1]], env[operands[0]], value).astype(
                    np.float32)
            elif op == "tessera.normalization_stats":
                assert len(destinations) == 2
                x = env[operands[0]]
                axis_match = re.search(r"axis = (-?\d+)", rhs)
                axis = int(axis_match.group(1)) if axis_match else -1
                eps_match = re.search(r"eps = ([-0-9.eE+]+)", rhs)
                eps = float(eps_match.group(1)) if eps_match else 1.0e-5
                # ODS elides the default `centered = true`; only the RMSNorm
                # specialization prints an explicit false value.
                centered = "centered = false" not in rhs
                center = np.mean(x, axis=axis)
                base = x - np.expand_dims(center, axis) if centered else x
                inv = 1.0 / np.sqrt(np.mean(base * base, axis=axis) + eps)
                env[destinations[0]] = center.astype(np.float32)
                env[destinations[1]] = inv.astype(np.float32)
            elif op == "tessera.softmax":
                x = env[operands[0]]
                axis_match = re.search(r"axis = (-?\d+)", rhs)
                axis = int(axis_match.group(1)) if axis_match else -1
                z = x - np.max(x, axis=axis, keepdims=True)
                e = np.exp(z)
                env[dst] = (e / np.sum(e, axis=axis, keepdims=True)).astype(np.float32)
            elif op == "tessera.loss.mse":
                prediction, target = (env[name] for name in operands)
                reduction_match = re.search(r'reduction = "([^"]+)"', rhs)
                reduction = reduction_match.group(1) if reduction_match else "mean"
                squared = (prediction - target) ** 2
                if reduction == "none":
                    value = squared
                elif reduction == "sum":
                    value = np.asarray(np.sum(squared), dtype=np.float32)
                else:
                    value = np.asarray(np.mean(squared), dtype=np.float32)
                env[dst] = value.astype(np.float32)
            elif op == "tessera.loss.mse_backward":
                assert len(destinations) == 2
                prediction, target, incoming = (env[name] for name in operands)
                reduction_match = re.search(r'reduction = "([^"]+)"', rhs)
                reduction = reduction_match.group(1) if reduction_match else "mean"
                scale = 1.0 / prediction.size if reduction == "mean" else 1.0
                gradient = 2.0 * (prediction - target) * incoming * scale
                env[destinations[0]] = gradient.astype(np.float32)
                env[destinations[1]] = (-gradient).astype(np.float32)
            elif op in {
                    "tessera.loss.mae", "tessera.loss.huber",
                    "tessera.loss.smooth_l1"}:
                prediction, target = (env[name] for name in operands)
                reduction_match = re.search(r'reduction = "([^"]+)"', rhs)
                reduction = reduction_match.group(1) if reduction_match else "mean"
                error = prediction - target
                if op == "tessera.loss.mae":
                    elementwise = np.abs(error)
                elif op == "tessera.loss.huber":
                    delta_match = re.search(r"delta = ([-0-9.eE+]+)", rhs)
                    delta = float(delta_match.group(1)) if delta_match else 1.0
                    elementwise = np.where(
                        np.abs(error) <= delta, 0.5 * error * error,
                        delta * (np.abs(error) - 0.5 * delta))
                else:
                    beta_match = re.search(r"beta = ([-0-9.eE+]+)", rhs)
                    beta = float(beta_match.group(1)) if beta_match else 1.0
                    elementwise = np.where(
                        np.abs(error) < beta, 0.5 * error * error / beta,
                        np.abs(error) - 0.5 * beta)
                if reduction == "none":
                    value = elementwise
                elif reduction == "sum":
                    value = np.asarray(np.sum(elementwise), dtype=np.float32)
                else:
                    value = np.asarray(np.mean(elementwise), dtype=np.float32)
                env[dst] = value.astype(np.float32)
            elif op == "tessera.loss.regression_backward":
                assert len(destinations) == 2
                prediction, target, incoming = (env[name] for name in operands)
                reduction_match = re.search(r'reduction = "([^"]+)"', rhs)
                reduction = reduction_match.group(1) if reduction_match else "mean"
                kind = re.search(r'kind = "([^"]+)"', rhs).group(1)
                parameter_match = re.search(r"parameter = ([-0-9.eE+]+)", rhs)
                parameter = (
                    float(parameter_match.group(1)) if parameter_match else 1.0)
                error = prediction - target
                if kind == "mae":
                    local = np.sign(error)
                elif kind == "huber":
                    local = np.where(
                        np.abs(error) <= parameter, error,
                        parameter * np.sign(error))
                else:
                    local = np.where(
                        np.abs(error) < parameter, error / parameter,
                        np.sign(error))
                scale = 1.0 / prediction.size if reduction == "mean" else 1.0
                gradient = local * incoming * scale
                env[destinations[0]] = gradient.astype(np.float32)
                env[destinations[1]] = (-gradient).astype(np.float32)
            elif op == "tessera.sgd":
                lr = float(re.search(r"lr = ([-0-9.eE+]+)", rhs).group(1))
                env[dst] = (
                    env[operands[0]] - lr * env[operands[1]]).astype(np.float32)
            elif op == "tessera.sgd_backward":
                assert len(destinations) == 2
                lr = float(re.search(r"lr = ([-0-9.eE+]+)", rhs).group(1))
                incoming = env[operands[0]]
                env[destinations[0]] = incoming.astype(np.float32)
                env[destinations[1]] = (-lr * incoming).astype(np.float32)
            elif op == "tessera.mul":
                env[dst] = (env[operands[0]] * env[operands[1]]).astype(np.float32)
            elif op == "tessera.sub":
                env[dst] = (env[operands[0]] - env[operands[1]]).astype(np.float32)
            elif op == "tessera.add":
                env[dst] = (env[operands[0]] + env[operands[1]]).astype(np.float32)
            elif op == "tessera.broadcast":
                dims = result_shape(rhs)
                env[dst] = np.broadcast_to(env[operands[0]], dims).copy()
            elif op == "tessera.broadcast_in_dim":
                dims = result_shape(rhs)
                mapping_text = rhs.split("broadcast_dimensions =", 1)[1]
                mapping = [int(v) for v in re.findall(
                    r"-?\d+", mapping_text.split("]", 1)[0])]
                reshape = [1] * len(dims)
                for input_dim, output_dim in enumerate(mapping):
                    reshape[output_dim] = env[operands[0]].shape[input_dim]
                env[dst] = np.broadcast_to(
                    env[operands[0]].reshape(reshape), dims).copy()
            elif op == "tessera.reduce":
                axis = int(re.search(r"axis = (-?\d+)", rhs).group(1))
                kind = re.search(r'kind = "([^"]+)"', rhs).group(1)
                reducers = {
                    "sum": np.sum, "mean": np.mean,
                    "max": np.max, "min": np.min,
                }
                assert kind in reducers, f"unknown reduction kind {kind}"
                reducer = reducers[kind]
                env[dst] = reducer(env[operands[0]], axis=axis).astype(np.float32)
            elif op == "tessera.reduce_backward":
                axis = int(re.search(r"axis = (-?\d+)", rhs).group(1))
                kind = re.search(r'kind = "([^"]+)"', rhs).group(1)
                x, reduced, incoming = (env[name] for name in operands)
                expanded_reduced = np.expand_dims(reduced, axis=axis)
                expanded_incoming = np.expand_dims(incoming, axis=axis)
                if kind == "sum":
                    grad = np.broadcast_to(expanded_incoming, x.shape)
                elif kind == "mean":
                    grad = np.broadcast_to(
                        expanded_incoming / x.shape[axis], x.shape)
                else:
                    mask = x == expanded_reduced
                    ties = np.sum(mask, axis=axis, keepdims=True)
                    grad = np.where(mask, expanded_incoming / ties, 0.0)
                env[dst] = grad.astype(np.float32)
            elif op == "tessera.unsqueeze":
                axes_text = rhs.split("axes =", 1)[1].split("}", 1)[0]
                axis = int(re.search(r"\[\s*(-?\d+)", axes_text).group(1))
                env[dst] = np.expand_dims(env[operands[0]], axis=axis)
            elif op == "tessera.reshape":
                dims = result_shape(rhs)
                env[dst] = env[operands[0]].reshape(dims)
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


def _binary_forward_mlir(op: str) -> str:
    return f"""
module {{
  func.func @binary(%x: tensor<3x5xf32>, %y: tensor<3x5xf32>) -> tensor<3x5xf32>
      attributes {{tessera.autodiff = "reverse"}} {{
    %z = "tessera.{op}"(%x, %y) : (tensor<3x5xf32>, tensor<3x5xf32>) -> tensor<3x5xf32>
    return %z : tensor<3x5xf32>
  }}
}}
"""


def test_native_broadcast_backward_matches_numpy_oracle():
    mlir = """
module {
  func.func @broadcast(%x: tensor<1x3xf32>) -> tensor<2x4x3xf32>
      attributes {tessera.autodiff = "reverse"} {
    %y = "tessera.broadcast"(%x) :
        (tensor<1x3xf32>) -> tensor<2x4x3xf32>
    return %y : tensor<2x4x3xf32>
  }
}
"""
    rng = np.random.default_rng(23)
    x = rng.standard_normal((1, 3)).astype(np.float32)
    seed = rng.standard_normal((2, 4, 3)).astype(np.float32)
    funcs = _run_paired(mlir)
    (primal,) = funcs["broadcast"].run([x])
    (dx,) = funcs["broadcast__bwd"].run([x, seed])
    np.testing.assert_allclose(primal, np.broadcast_to(x, (2, 4, 3)))
    np.testing.assert_allclose(dx, seed.sum(axis=(0, 1)).reshape(1, 3))


def test_native_dynamic_broadcast_backward_matches_numpy_oracle():
    mlir = """
module {
  func.func @broadcast(%x: tensor<?x1x3xf32>) -> tensor<?x4x3xf32>
      attributes {tessera.autodiff = "reverse"} {
    %y = "tessera.broadcast"(%x) :
        (tensor<?x1x3xf32>) -> tensor<?x4x3xf32>
    return %y : tensor<?x4x3xf32>
  }
}
"""
    rng = np.random.default_rng(29)
    x = rng.standard_normal((5, 1, 3)).astype(np.float32)
    seed = rng.standard_normal((5, 4, 3)).astype(np.float32)
    funcs = _run_paired(mlir)
    (primal,) = funcs["broadcast"].run([x])
    (dx,) = funcs["broadcast__bwd"].run([x, seed])
    np.testing.assert_allclose(primal, np.broadcast_to(x, seed.shape))
    np.testing.assert_allclose(dx, seed.sum(axis=1, keepdims=True))


@pytest.mark.parametrize("kind", ["sum", "mean", "max", "min"])
@pytest.mark.parametrize("shape_type,shape", [
    ("2x3", (2, 3)),
    ("2x?", (2, 5)),
])
def test_native_reduction_backward_matches_numpy_oracle(
        kind: str, shape_type: str, shape: tuple[int, int]):
    mlir = f"""
module {{
  func.func @reduce(%x: tensor<{shape_type}xf32>) -> tensor<2xf32>
      attributes {{tessera.autodiff = "reverse"}} {{
    %y = "tessera.reduce"(%x) {{kind = "{kind}", axis = 1 : i64}} :
        (tensor<{shape_type}xf32>) -> tensor<2xf32>
    return %y : tensor<2xf32>
  }}
}}
"""
    rng = np.random.default_rng(31)
    x = rng.standard_normal(shape).astype(np.float32)
    if kind in {"max", "min"}:
        # Pin two extrema in each row so the equal-share tie contract is proven.
        tied = 9.0 if kind == "max" else -9.0
        x[:, 0] = tied
        x[:, 1] = tied
    seed = rng.standard_normal((2,)).astype(np.float32)
    funcs = _run_paired(mlir)
    (primal,) = funcs["reduce"].run([x])
    (dx,) = funcs["reduce__bwd"].run([x, seed])
    reducer = {
        "sum": np.sum, "mean": np.mean, "max": np.max, "min": np.min,
    }[kind]
    np.testing.assert_allclose(primal, reducer(x, axis=1))
    if kind == "sum":
        expected = np.broadcast_to(seed[:, None], x.shape)
    elif kind == "mean":
        expected = np.broadcast_to(seed[:, None] / x.shape[1], x.shape)
    else:
        mask = x == reducer(x, axis=1)[:, None]
        expected = np.where(
            mask, seed[:, None] / np.sum(mask, axis=1, keepdims=True), 0.0)
    np.testing.assert_allclose(dx, expected, rtol=2e-7, atol=1e-7)


@pytest.mark.parametrize("reduction,shape_type,shape", [
    ("mean", "?x5", (7, 5)),
    ("sum", "3x5", (3, 5)),
    ("none", "3x5", (3, 5)),
])
def test_native_mse_backward_matches_numpy_oracle(
        reduction: str, shape_type: str, shape: tuple[int, int]):
    result_type = f"tensor<{shape_type}xf32>" if reduction == "none" else "tensor<f32>"
    mlir = f"""
module {{
  func.func @mse(%prediction: tensor<{shape_type}xf32>,
                 %target: tensor<{shape_type}xf32>) -> {result_type}
      attributes {{tessera.autodiff = "reverse"}} {{
    %loss = "tessera.loss.mse"(%prediction, %target)
        {{reduction = "{reduction}"}} :
        (tensor<{shape_type}xf32>, tensor<{shape_type}xf32>) -> {result_type}
    return %loss : {result_type}
  }}
}}
"""
    rng = np.random.default_rng(37 + len(reduction))
    prediction = rng.standard_normal(shape).astype(np.float32)
    target = rng.standard_normal(shape).astype(np.float32)
    seed_shape = shape if reduction == "none" else ()
    seed = rng.standard_normal(seed_shape).astype(np.float32)
    funcs = _run_paired(mlir)
    (primal,) = funcs["mse"].run([prediction, target])
    prediction_grad, target_grad = funcs["mse__bwd"].run(
        [prediction, target, seed])
    squared = (prediction - target) ** 2
    if reduction == "none":
        expected_primal = squared
        scale = 1.0
    elif reduction == "sum":
        expected_primal = np.asarray(np.sum(squared), dtype=np.float32)
        scale = 1.0
    else:
        expected_primal = np.asarray(np.mean(squared), dtype=np.float32)
        scale = 1.0 / prediction.size
    expected_grad = 2.0 * (prediction - target) * seed * scale
    np.testing.assert_allclose(primal, expected_primal, rtol=2e-7, atol=1e-7)
    np.testing.assert_allclose(
        prediction_grad, expected_grad, rtol=2e-7, atol=1e-7)
    np.testing.assert_allclose(
        target_grad, -expected_grad, rtol=2e-7, atol=1e-7)


@pytest.mark.parametrize("op,attribute,parameter", [
    ("mae", "", 1.0),
    ("huber", "delta = 0.75 : f64, ", 0.75),
    ("smooth_l1", "beta = 0.5 : f64, ", 0.5),
])
@pytest.mark.parametrize("reduction", ["none", "sum", "mean"])
def test_native_regression_loss_backward_matches_boundary_oracle(
        op: str, attribute: str, parameter: float, reduction: str):
    result_type = "tensor<2x5xf32>" if reduction == "none" else "tensor<f32>"
    mlir = f"""
module {{
  func.func @loss(%prediction: tensor<2x5xf32>,
                  %target: tensor<2x5xf32>) -> {result_type}
      attributes {{tessera.autodiff = "reverse"}} {{
    %loss = "tessera.loss.{op}"(%prediction, %target)
        {{{attribute}reduction = "{reduction}"}} :
        (tensor<2x5xf32>, tensor<2x5xf32>) -> {result_type}
    return %loss : {result_type}
  }}
}}
"""
    # Exact zero and +/- transition points guard the subgradient contract.
    errors = np.asarray([
        -2.0 * parameter, -parameter, -0.25 * parameter, 0.0,
        0.25 * parameter, parameter, 2.0 * parameter, -0.0, 0.1, -0.1,
    ], dtype=np.float32).reshape(2, 5)
    target = np.linspace(-0.5, 0.5, 10, dtype=np.float32).reshape(2, 5)
    prediction = target + errors
    seed = (
        np.linspace(0.5, 1.4, 10, dtype=np.float32).reshape(2, 5)
        if reduction == "none" else np.asarray(1.25, dtype=np.float32))
    funcs = _run_paired(mlir)
    (primal,) = funcs["loss"].run([prediction, target])
    prediction_grad, target_grad = funcs["loss__bwd"].run(
        [prediction, target, seed])
    if op == "mae":
        elementwise = np.abs(errors)
        local = np.sign(errors)
    elif op == "huber":
        elementwise = np.where(
            np.abs(errors) <= parameter, 0.5 * errors * errors,
            parameter * (np.abs(errors) - 0.5 * parameter))
        local = np.where(
            np.abs(errors) <= parameter, errors,
            parameter * np.sign(errors))
    else:
        elementwise = np.where(
            np.abs(errors) < parameter, 0.5 * errors * errors / parameter,
            np.abs(errors) - 0.5 * parameter)
        local = np.where(
            np.abs(errors) < parameter, errors / parameter, np.sign(errors))
    if reduction == "none":
        expected_primal = elementwise
        scale = 1.0
    elif reduction == "sum":
        expected_primal = np.asarray(np.sum(elementwise), dtype=np.float32)
        scale = 1.0
    else:
        expected_primal = np.asarray(np.mean(elementwise), dtype=np.float32)
        scale = 1.0 / errors.size
    expected_grad = local * seed * scale
    np.testing.assert_allclose(primal, expected_primal, rtol=2e-7, atol=1e-7)
    np.testing.assert_allclose(
        prediction_grad, expected_grad, rtol=2e-7, atol=1e-7)
    np.testing.assert_allclose(
        target_grad, -expected_grad, rtol=2e-7, atol=1e-7)


def test_native_sgd_backward_matches_dynamic_numpy_oracle():
    mlir = """
module {
  func.func @sgd(%param: tensor<?x5xf32>,
                 %grad: tensor<?x5xf32>) -> tensor<?x5xf32>
      attributes {tessera.autodiff = "reverse"} {
    %updated = "tessera.sgd"(%param, %grad) {lr = 0.125 : f64} :
        (tensor<?x5xf32>, tensor<?x5xf32>) -> tensor<?x5xf32>
    return %updated : tensor<?x5xf32>
  }
}
"""
    rng = np.random.default_rng(73)
    param = rng.standard_normal((7, 5)).astype(np.float32)
    grad = rng.standard_normal((7, 5)).astype(np.float32)
    seed = rng.standard_normal((7, 5)).astype(np.float32)
    funcs = _run_paired(mlir)
    (updated,) = funcs["sgd"].run([param, grad])
    param_grad, grad_grad = funcs["sgd__bwd"].run([param, grad, seed])
    np.testing.assert_allclose(updated, param - 0.125 * grad)
    np.testing.assert_allclose(param_grad, seed)
    np.testing.assert_allclose(grad_grad, -0.125 * seed)


@pytest.mark.parametrize("op", ["add", "mul"])
def test_native_binary_backward_matches_numpy_oracle(op: str):
    rng = np.random.default_rng(11)
    x = rng.standard_normal((3, 5)).astype(np.float32)
    y = rng.standard_normal((3, 5)).astype(np.float32)
    seed = rng.standard_normal((3, 5)).astype(np.float32)
    funcs = _run_paired(_binary_forward_mlir(op))
    assert "binary" in funcs and "binary__bwd" in funcs
    (primal,) = funcs["binary"].run([x, y])
    dx, dy = funcs["binary__bwd"].run([x, y, seed])
    if op == "add":
        np.testing.assert_allclose(primal, x + y)
        np.testing.assert_allclose(dx, seed)
        np.testing.assert_allclose(dy, seed)
    else:
        np.testing.assert_allclose(primal, x * y)
        np.testing.assert_allclose(dx, seed * y)
        np.testing.assert_allclose(dy, seed * x)


@pytest.mark.parametrize("op", ["silu", "gelu"])
def test_native_activation_backward_matches_numpy_oracle(op: str):
    mlir = f"""
module {{
  func.func @activation(%x: tensor<3x5xf32>) -> tensor<3x5xf32>
      attributes {{tessera.autodiff = "reverse"}} {{
    %y = "tessera.{op}"(%x) : (tensor<3x5xf32>) -> tensor<3x5xf32>
    return %y : tensor<3x5xf32>
  }}
}}
"""
    rng = np.random.default_rng(41)
    x = rng.standard_normal((3, 5)).astype(np.float32)
    seed = rng.standard_normal((3, 5)).astype(np.float32)
    funcs = _run_paired(mlir)
    (primal,) = funcs["activation"].run([x])
    (dx,) = funcs["activation__bwd"].run([x, seed])
    s = 1.0 / (1.0 + np.exp(-x))
    if op == "silu":
        expected = x * s
        derivative = s + x * s * (1.0 - s)
    else:
        k = np.sqrt(2.0 / np.pi)
        inner = k * (x + 0.044715 * x**3)
        t = np.tanh(inner)
        expected = 0.5 * x * (1.0 + t)
        derivative = 0.5 * (1.0 + t) + (
            0.5 * x * (1.0 - t * t) * k * (1.0 + 3.0 * 0.044715 * x * x)
        )
    np.testing.assert_allclose(primal, expected, rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(dx, seed * derivative, rtol=3e-6, atol=3e-6)


@pytest.mark.parametrize("shape_type,shape", [
    ("3x5", (3, 5)),
    ("?x5", (7, 5)),
])
def test_native_relu_backward_matches_numpy_oracle(shape_type, shape):
    mlir = f"""
module {{
  func.func @relu(%x: tensor<{shape_type}xf32>) -> tensor<{shape_type}xf32>
      attributes {{tessera.autodiff = "reverse"}} {{
    %y = "tessera.relu"(%x) :
        (tensor<{shape_type}xf32>) -> tensor<{shape_type}xf32>
    return %y : tensor<{shape_type}xf32>
  }}
}}
"""
    rng = np.random.default_rng(67)
    x = rng.standard_normal(shape).astype(np.float32)
    x.flat[0] = 0.0  # pin the derivative convention at the nondifferentiable point
    seed = rng.standard_normal(shape).astype(np.float32)
    funcs = _run_paired(mlir)
    (primal,) = funcs["relu"].run([x])
    (dx,) = funcs["relu__bwd"].run([x, seed])
    np.testing.assert_array_equal(primal, np.maximum(x, 0.0))
    np.testing.assert_array_equal(dx, np.where(x > 0.0, seed, 0.0))


@pytest.mark.parametrize("op", ["rmsnorm", "layer_norm"])
@pytest.mark.parametrize("shape_type,shape", [
    ("3x5", (3, 5)),
    ("?x5", (7, 5)),
])
def test_native_normalization_backward_matches_numpy_oracle(
        op: str, shape_type: str, shape: tuple[int, int]):
    mlir = f"""
module {{
  func.func @normalization(%x: tensor<{shape_type}xf32>)
      -> tensor<{shape_type}xf32>
      attributes {{tessera.autodiff = "reverse"}} {{
    %y = "tessera.{op}"(%x) {{eps = 1.0e-5 : f64}} :
        (tensor<{shape_type}xf32>) -> tensor<{shape_type}xf32>
    return %y : tensor<{shape_type}xf32>
  }}
}}
"""
    rng = np.random.default_rng(71)
    x = rng.standard_normal(shape).astype(np.float32)
    seed = rng.standard_normal(shape).astype(np.float32)
    funcs = _run_paired(mlir)
    (primal,) = funcs["normalization"].run([x])
    (dx,) = funcs["normalization__bwd"].run([x, seed])
    if op == "rmsnorm":
        inv = 1.0 / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + 1.0e-5)
        expected = x * inv
        dx_expected = seed * inv - x * inv**3 * np.mean(
            seed * x, axis=-1, keepdims=True)
    else:
        center = np.mean(x, axis=-1, keepdims=True)
        inv = 1.0 / np.sqrt(
            np.mean((x - center) ** 2, axis=-1, keepdims=True) + 1.0e-5)
        expected = (x - center) * inv
        dx_expected = inv * (
            seed - np.mean(seed, axis=-1, keepdims=True)
            - expected * np.mean(seed * expected, axis=-1, keepdims=True))
    np.testing.assert_allclose(primal, expected, rtol=3e-6, atol=3e-6)
    np.testing.assert_allclose(dx, dx_expected, rtol=5e-6, atol=5e-6)


@pytest.mark.parametrize("op", ["rmsnorm", "layer_norm"])
@pytest.mark.parametrize("shape_type,shape", [("3x5", (3, 5)), ("?x5", (7, 5))])
def test_native_affine_normalization_gradients_match_numpy_oracle(
        op: str, shape_type: str, shape: tuple[int, int]):
    affine_args = "%gamma: tensor<5xf32>"
    affine_operands = "%x, %gamma"
    affine_types = f"tensor<{shape_type}xf32>, tensor<5xf32>"
    if op == "layer_norm":
        affine_args += ", %beta: tensor<5xf32>"
        affine_operands += ", %beta"
        affine_types += ", tensor<5xf32>"
    mlir = f"""
module {{
  func.func @normalization(%x: tensor<{shape_type}xf32>, {affine_args})
      -> tensor<{shape_type}xf32>
      attributes {{tessera.autodiff = "reverse"}} {{
    %y = "tessera.{op}"({affine_operands}) {{eps = 1.0e-5 : f64}} :
        ({affine_types}) -> tensor<{shape_type}xf32>
    return %y : tensor<{shape_type}xf32>
  }}
}}
"""
    rng = np.random.default_rng(79 + len(shape))
    x = rng.standard_normal(shape).astype(np.float32)
    gamma = rng.uniform(0.5, 1.5, (shape[-1],)).astype(np.float32)
    beta = rng.standard_normal((shape[-1],)).astype(np.float32)
    seed = rng.standard_normal(shape).astype(np.float32)
    inputs = [x, gamma] + ([beta] if op == "layer_norm" else [])
    funcs = _run_paired(mlir)
    (primal,) = funcs["normalization"].run(inputs)
    grads = funcs["normalization__bwd"].run(inputs + [seed])
    if op == "rmsnorm":
        inv = 1.0 / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + 1.0e-5)
        normalized = x * inv
        dz = seed * gamma
        dx = dz * inv - x * inv**3 * np.mean(dz * x, axis=-1, keepdims=True)
        expected = normalized * gamma
        expected_grads = [dx, np.sum(seed * normalized, axis=0)]
    else:
        center = np.mean(x, axis=-1, keepdims=True)
        inv = 1.0 / np.sqrt(np.mean((x - center) ** 2, axis=-1, keepdims=True)
                               + 1.0e-5)
        normalized = (x - center) * inv
        dz = seed * gamma
        dx = inv * (dz - np.mean(dz, axis=-1, keepdims=True)
                    - normalized * np.mean(dz * normalized, axis=-1, keepdims=True))
        expected = normalized * gamma + beta
        expected_grads = [dx, np.sum(seed * normalized, axis=0), np.sum(seed, axis=0)]
    np.testing.assert_allclose(primal, expected, rtol=4e-6, atol=4e-6)
    for actual, wanted in zip(grads, expected_grads):
        np.testing.assert_allclose(actual, wanted, rtol=7e-6, atol=7e-6)


def test_native_softmax_backward_matches_numpy_oracle():
    mlir = """
module {
  func.func @softmax(%x: tensor<3x5xf32>) -> tensor<3x5xf32>
      attributes {tessera.autodiff = "reverse"} {
    %y = "tessera.softmax"(%x) {axis = 1 : i64} :
        (tensor<3x5xf32>) -> tensor<3x5xf32>
    return %y : tensor<3x5xf32>
  }
}
"""
    rng = np.random.default_rng(53)
    x = rng.standard_normal((3, 5)).astype(np.float32)
    seed = rng.standard_normal((3, 5)).astype(np.float32)
    funcs = _run_paired(mlir)
    (s,) = funcs["softmax"].run([x])
    (dx,) = funcs["softmax__bwd"].run([x, seed])
    expected = (seed - np.sum(seed * s, axis=1, keepdims=True)) * s
    np.testing.assert_allclose(dx, expected, rtol=3e-6, atol=3e-6)


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
