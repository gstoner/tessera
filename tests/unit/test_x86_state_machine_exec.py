"""W4-PRODUCT-1 — the bounded IRREDUCIBLE-CFG state machine executes
natively on x86 (AVX-512 host), forward and generated backward: the sibling
row to the gfx1151 one (sync key `W4-SM-ROCM-2026-08-21`).

The chain is the compiler's own output end to end — no reference
re-implementation and no hand-assembled lowering:

  source two-entry-SCC CFG → tessera-opt --tessera-autodiff-paired
  (structurizes to the typed `bounded_state_machine_v1` program-counter
  machine + generates the recompute_all backward) → tessera_jit
  (tessera-to-linalg → convert-elementwise-to-linalg → one-shot-bufferize
  → loops → LLVM → ORC JIT) → native call through the memref C ABI.

Both entry paths of the irreducible SCC are exercised by specializing the
entry flag as a body constant — the CFG keeps BOTH entry edges into the
bb1/bb2 cycle, so the Tarjan classification still sees a two-entry
irreducible SCC (asserted: the paired output carries
`bounded_state_machine_v1`); only the runtime path through it is pinned:

  enter_left=true : entry→bb1→bb2→bb1→bb2→bb3  ⇒ y = tanh(tanh(x))
  enter_left=false: entry→bb2→bb1→bb2→bb3      ⇒ y = tanh(x)

The row binds the exact CFG identity: the module handed to the JIT is
asserted to carry the structured-CFG digest and the residual policy, and
the proof-of-execution counter distinguishes a real JIT run from any
fallback. Unlike the ROCm lane's host-checked STATUS buffer, `cf.assert`
compiles NATIVELY here (ControlFlowToLLVM), so max_steps exhaustion traps
in-process — the stronger form of the same bound enforcement.

A second machine pins the interior vocabulary the #606 review named:
per-element data-dependent selection (`arith.cmpf` + `arith.select` over
the tensor slots) through the same native chain.

Correctness-only rows (WSL — Decision #26a timing rules). Skip-clean when
tessera-opt or libtessera_jit is not built.
"""

from __future__ import annotations

import numpy as np
import pytest

from tessera import _jit_boundary as jb
from tests._support.compiler_tool import require_tessera_opt, run_tessera_opt

pytestmark = pytest.mark.skipif(
    not jb.is_available(),
    reason="libtessera_jit not built; run `ninja -C build tessera_jit`")

DIGEST = "9" * 64
SELECT_DIGEST = "a" * 64


def _irreducible_mlir(n: int, enter_left: bool) -> str:
    flag = "true" if enter_left else "false"
    return f"""
module {{
  func.func @irreducible(%x: tensor<{n}xf32>)
      -> tensor<{n}xf32> attributes {{tessera.autodiff = "reverse"}} {{
    %out = scf.execute_region -> tensor<{n}xf32> {{
      %c0 = arith.constant 0 : index
      %enter_left = arith.constant {flag}
      cf.cond_br %enter_left, ^bb1(%c0, %x : index, tensor<{n}xf32>),
                              ^bb2(%c0, %x : index, tensor<{n}xf32>)
    ^bb1(%i: index, %state: tensor<{n}xf32>):
      %next = "tessera.tanh"(%state) :
          (tensor<{n}xf32>) -> tensor<{n}xf32>
      cf.br ^bb2(%i, %next : index, tensor<{n}xf32>)
    ^bb2(%j: index, %right_state: tensor<{n}xf32>):
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %next_i = arith.addi %j, %c1 : index
      %continue = arith.cmpi slt, %next_i, %c2 : index
      cf.cond_br %continue,
          ^bb1(%next_i, %right_state : index, tensor<{n}xf32>),
          ^bb3(%right_state : tensor<{n}xf32>)
    ^bb3(%result: tensor<{n}xf32>):
      scf.yield %result : tensor<{n}xf32>
    }} {{tessera.structured_cfg.digest = "{DIGEST}",
       tessera.structured_cfg.max_steps = 8 : i64}}
    return %out : tensor<{n}xf32>
  }}
}}
"""


def _paired_state_machine(src: str, digest: str) -> str:
    require_tessera_opt()
    paired = run_tessera_opt(src, "--tessera-autodiff-paired")
    assert paired.returncode == 0, f"paired failed: {paired.stderr}"
    # The structurizer must have classified this as the bounded state
    # machine (a reducible collapse would be a different, weaker claim) and
    # the module the JIT compiles must BIND the CFG identity + residual
    # policy — that is what makes this an execution row, not a smoke test.
    assert 'tessera.structured_cfg.execution = "bounded_state_machine_v1"' \
        in paired.stdout
    assert f'tessera.structured_cfg.digest = "{digest}"' in paired.stdout
    return paired.stdout


@pytest.mark.parametrize("enter_left", [True, False])
def test_irreducible_state_machine_executes_natively_on_x86(enter_left):
    n = 300
    module = _paired_state_machine(_irreducible_mlir(n, enter_left), DIGEST)
    assert 'tessera.autodiff.residual_policy = "recompute_all"' in module
    handle = jb.compile_module(module)

    rng = np.random.default_rng(5)
    x = rng.standard_normal(n).astype(np.float32)
    dout = rng.standard_normal(n).astype(np.float32)
    t1 = np.tanh(x)
    t2 = np.tanh(t1)

    n0 = jb.invocation_count()
    out = np.zeros(n, dtype=np.float32)
    jb.invoke(handle, "irreducible", [x], out)
    ref_fwd = t2 if enter_left else t1
    np.testing.assert_allclose(out, ref_fwd, rtol=1e-6, atol=1e-7)

    dx = np.zeros(n, dtype=np.float32)
    jb.invoke(handle, "irreducible__bwd", [x, dout], dx)
    ref_bwd = (dout * (1.0 - t2 * t2) * (1.0 - t1 * t1) if enter_left
               else dout * (1.0 - t1 * t1))
    np.testing.assert_allclose(dx, ref_bwd, rtol=1e-5, atol=1e-6)

    # Proof of execution: both calls went through the native JIT, not any
    # fallback path.
    assert jb.invocation_count() - n0 == 2


def test_data_dependent_select_machine_executes_natively_on_x86():
    """Interior vocabulary (the #606 review's P2, x86 side): a machine whose
    state update is per-element data-dependent — three steps of
    select(s > 0, tanh(s), s) — through the same native chain."""
    n = 300
    src = f"""
module {{
  func.func @select_machine(%x: tensor<{n}xf32>) -> tensor<{n}xf32> {{
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %r = scf.for %i = %c0 to %c3 step %c1 iter_args(%s = %x)
        -> (tensor<{n}xf32>) {{
      %zero = arith.constant dense<0.0> : tensor<{n}xf32>
      %p = arith.cmpf ogt, %s, %zero : tensor<{n}xf32>
      %t = "tessera.tanh"(%s) : (tensor<{n}xf32>) -> tensor<{n}xf32>
      %n2 = arith.select %p, %t, %s : tensor<{n}xi1>, tensor<{n}xf32>
      scf.yield %n2 : tensor<{n}xf32>
    }} {{tessera.structured_cfg.execution = "bounded_state_machine_v1",
       tessera.structured_cfg.digest = "{SELECT_DIGEST}",
       tessera.structured_cfg.max_steps = 4 : i64}}
    return %r : tensor<{n}xf32>
  }}
}}
"""
    require_tessera_opt()
    handle = jb.compile_module(src)
    rng = np.random.default_rng(9)
    x = rng.standard_normal(n).astype(np.float32)
    n0 = jb.invocation_count()
    out = np.zeros(n, dtype=np.float32)
    jb.invoke(handle, "select_machine", [x], out)
    assert jb.invocation_count() - n0 == 1
    ref = x.copy()
    for _ in range(3):
        ref = np.where(ref > 0.0, np.tanh(ref), ref)
    np.testing.assert_allclose(out, ref, rtol=1e-6, atol=1e-7)
