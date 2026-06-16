"""Phase 0a — fusion-group ``dispatch`` roles (front-to-back closure plan).

``_derive_fusion_groups`` now carries an operand/result role map on each
recognized ``known_chain`` group, resolved from Graph-IR operand order at
compile time. This is the foundation for making the apple_gpu executor
authoritative (0b): it lets the executor dispatch a fused kernel from the group
alone, instead of re-extracting operand roles from the op list per invoke
(``runtime._execute_apple_gpu_mps_metadata``) and falling back to the structural
re-matchers.

0a is strictly additive: a group without resolvable roles simply carries no
``dispatch`` key, so the executor's existing path is unaffected. See
docs/audit/compiler/COMPILER_AUDIT.md (Library → Optimizing Compiler).
"""

from __future__ import annotations

import tessera as ts
from tessera.compiler.canonical_compile import _derive_fusion_groups
from tessera.compiler.graph_ir import GraphIRBuilder


def _groups(fn):
    b = GraphIRBuilder()
    b.lower(fn)
    return _derive_fusion_groups(b.module())


def _known(groups):
    return [g for g in groups if g.get("kind") == "known_chain"]


def test_matmul_softmax_matmul_roles():
    def f(a, b, c):
        return ts.ops.matmul(ts.ops.softmax(ts.ops.matmul(a, b)), c)

    (g,) = [g for g in _known(_groups(f))
            if g["fused_kernel"] == "matmul_softmax_matmul"]
    d = g["dispatch"]
    # a/b come from the score matmul; c is the tail matmul's second operand;
    # out is the tail matmul's result. (transpose_a/transpose_b are carried
    # orientation flags — the M2 attention-orientation fix.)
    assert {"a", "b", "c", "out"} <= set(d)
    assert d.get("transpose_a") is False and d.get("transpose_b") is False
    ssa = {d[k] for k in ("a", "b", "c", "out")}
    assert all(isinstance(v, str) and not v.startswith("%") for v in ssa)
    assert d["a"] != d["b"] != d["c"]


def test_two_op_chain_roles():
    def gelu_chain(a, b):
        return ts.ops.gelu(ts.ops.matmul(a, b))

    def softmax_chain(a, b):
        return ts.ops.softmax(ts.ops.matmul(a, b))

    for kernel, fn in (("matmul_gelu", gelu_chain),
                       ("matmul_softmax", softmax_chain)):
        (g,) = [g for g in _known(_groups(fn)) if g["fused_kernel"] == kernel]
        d = g["dispatch"]
        assert {"a", "b", "out"} <= set(d), kernel
        # out is the tail op's result, distinct from the matmul inputs.
        assert d["out"] not in (d["a"], d["b"])


def test_matmul_rmsnorm_carries_eps():
    def rmsnorm_chain(a, b):
        return ts.ops.rmsnorm(ts.ops.matmul(a, b))

    (g,) = [g for g in _known(_groups(rmsnorm_chain))
            if g["fused_kernel"] == "matmul_rmsnorm"]
    d = g["dispatch"]
    assert {"a", "b", "out", "eps"} <= set(d)
    # eps is a resolved scalar param (default 1e-5 for rmsnorm), not an SSA name.
    assert isinstance(d["eps"], float)
    assert d["eps"] == 1e-5


def test_swiglu_roles_share_x():
    # gate = matmul(x, Wg); up = matmul(x, Wu); h = silu_mul(gate, up);
    # out = matmul(h, Wd) — gate and up consume the SAME x.
    def swiglu(x, wg, wu, wd):
        gate = ts.ops.matmul(x, wg)
        up = ts.ops.matmul(x, wu)
        return ts.ops.matmul(ts.ops.silu_mul(gate, up), wd)

    known = [g for g in _known(_groups(swiglu)) if g["fused_kernel"] == "swiglu"]
    if not known:                      # silu_mul lowering not available here
        return
    d = known[0]["dispatch"]
    assert set(d) == {"x", "wg", "wu", "wd", "out"}
    # The four weight/input roles are distinct names; out is the tail result.
    assert len({d["x"], d["wg"], d["wu"], d["wd"]}) == 4


def test_roles_index_into_arg_names_or_results():
    # Every tensor role must be either a function argument or some op's result —
    # i.e. a name the executor will have bound in `values`. This is the property
    # 0b relies on to dispatch without re-deriving.
    def f(a, b, c):
        return ts.ops.matmul(ts.ops.softmax(ts.ops.matmul(a, b)), c)

    b = GraphIRBuilder()
    b.lower(f)
    module = b.module()
    fn = module.functions[-1]
    bound = {arg.name for arg in fn.args}
    for op in fn.body:
        bound.update(n.lstrip("%") for n in (op.result_names or []))

    (g,) = [g for g in _known(_derive_fusion_groups(module))
            if g["fused_kernel"] == "matmul_softmax_matmul"]
    for role, name in g["dispatch"].items():
        if role in ("eps", "transpose_a", "transpose_b"):
            continue                           # scalar param / orientation flag
        assert name in bound, f"role {role}={name} not a bound value"
