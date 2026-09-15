"""The matmul epilogue survives both frontends and the shared tiling pass.

`ops.matmul(a, w, bias=b, activation="gelu", residual=r)` reaches Graph IR as
a four-operand `tessera.matmul` whose epilogue operands are marked by the
string attributes `MatmulOp::verify` reads (`bias = "bias"`,
`residual = "residual"`) and whose activation rides as an attribute. The
shared `tessera-tiling` pass then tiles the plain product and re-applies the
epilogue once, in the contract's order (bias, activation, residual). Before
2026-09-15 the tracer emitted the operands without the markers, so the op
failed the C++ verifier and no frontend ever reached the tiling rewrite.
"""
from __future__ import annotations

import subprocess

import numpy as np
import pytest

from tests._support.environment import CompilerToolchain


def _traced_text(bias: bool, activation: str, residual: bool) -> str:
    from tessera import ops
    from tessera.compiler.trace import to_graph_ir_module, trace
    a = np.zeros((32, 64), np.float32)
    w = np.zeros((64, 48), np.float32)
    args: list = [a, w]
    names = ["a", "w"]
    kwargs: dict = {}
    if bias:
        args.append(np.zeros((48,), np.float32)); names.append("bias")
    if residual:
        args.append(np.zeros((32, 48), np.float32)); names.append("res")
    if activation != "none":
        kwargs["activation"] = activation

    def fn(*xs):
        call = {}
        i = 2
        if bias:
            call["bias"] = xs[i]; i += 1
        if residual:
            call["residual"] = xs[i]
        return ops.matmul(xs[0], xs[1], **call, **kwargs)

    traced = trace(fn, *args, arg_names=tuple(names))
    return to_graph_ir_module(traced, name="f").to_mlir(canonical=True)


@pytest.mark.parametrize("bias,activation,residual", [
    (True, "none", False), (True, "gelu", False), (True, "gelu", True), (False, "relu", True), (False, "none", True),
])
def test_tracer_marks_the_epilogue_operands_the_verifier_reads(bias, activation, residual):
    text = _traced_text(bias, activation, residual)
    line = next(l for l in text.splitlines() if "tessera.matmul" in l)
    assert ('bias = "bias"' in line) is bias
    assert ('residual = "residual"' in line) is residual
    assert (f'activation = "{activation}"' in line) is (activation != "none")
    operands = line.split("tessera.matmul ", 1)[1].split(" {", 1)[0].split(", ")
    assert len(operands) == 2 + int(bias) + int(residual)  # operands, not attributes


def test_ast_frontend_marks_the_same_operands():
    """The AST frontend shares `apply_presence_flags`, so a jitted body emits
    the same markers (the frontend differential certificate depends on it)."""
    import tessera
    from tessera import ops

    @tessera.jit
    def f(a, w, b):
        return ops.matmul(a, w, bias=b)

    text = f.graph_ir.to_mlir()
    line = next(l for l in text.splitlines() if "tessera.matmul" in l)
    assert 'bias = "bias"' in line and "residual" not in line


@pytest.mark.parametrize("activation", ["none", "gelu"])
def test_traced_epilogue_matmul_tiles_in_contract_order(compiler_toolchain: CompilerToolchain, activation):
    text = _traced_text(True, activation, True)
    opt = compiler_toolchain.require_tessera_opt("tessera-tiling")
    proc = subprocess.run([str(opt), "-", "--tessera-tiling", "--allow-unregistered-dialect"],
                          input=text, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    order = [l.strip().split(" ")[2] for l in proc.stdout.splitlines()
             if l.strip().startswith("%") and any(
                 tok in l for tok in ("tessera.broadcast", "tessera.gelu", " tessera.add "))
             and "k_reduction_accumulate" not in l]
    expected = ["tessera.broadcast", "tessera.add"] + (["tessera.gelu"] if activation != "none" else []) + ["tessera.add"]
    assert order == expected, proc.stdout
    inner = next(l for l in proc.stdout.splitlines() if "canonical_k_step" in l)
    assert "bias" not in inner and "residual" not in inner and "activation" not in inner
