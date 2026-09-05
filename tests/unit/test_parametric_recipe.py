import os
from pathlib import Path

import numpy as np
import pytest
import tessera
from tessera.compiler.graph_ir import GraphIRBuilder
from tessera.compiler.loop_idioms import recognize_matmul_loop
from tessera.compiler.parametric_recipe import prepare_recipe, _optimize
from tessera.compiler.presburger import PresburgerConstraint, PresburgerSystem

ROOT = Path(__file__).resolve().parents[2]


def tool():
    path = Path(os.environ.get('TESSERA_OPT', ROOT / 'build/tools/tessera-opt/tessera-opt'))
    if not path.is_file():
        pytest.skip('native tessera-opt required')
    return str(path)


def product(a: tessera.Tensor['M', 'K', 'f32'], b: tessera.Tensor['K', 'N', 'f32']):
    c = np.zeros((a.shape[0], b.shape[1]), dtype=a.dtype)
    for i in range(a.shape[0]):
        for j in range(b.shape[1]):
            for k in range(a.shape[1]):
                c[i, j] += a[i, k] * b[k, j]
    return c


def test_loop_candidate_oracle_and_shared_recipe():
    candidate = recognize_matmul_loop(product)
    rng = np.random.default_rng(7)
    for shape in [(2, 3, 4), (3, 4, 2)]:
        m, k, n = shape
        a = rng.normal(size=(m, k)).astype('f4')
        b = rng.normal(size=(k, n)).astype('f4')
        np.testing.assert_allclose(product(a, b), np.matmul(a, b), atol=1e-6, rtol=1e-6)
    system = PresburgerSystem(('K',), (PresburgerConstraint('mod', (1,), modulus=2),))
    recipe = candidate.prepare(tessera_opt=tool(), system=system)
    ranks = recipe.rank_buckets([{'M': 2, 'K': 4, 'N': 3}, {'M': 3, 'K': 3, 'N': 2}])
    assert [r.retained for r in ranks] == [True, False]
    assert len({r.recipe_digest for r in ranks}) == 1
    assert all(not r.promotion_eligible for r in ranks)
    assert 'tensor<?x?xf32>' in recipe.optimized_mlir
    assert 'tessera.matmul' in recipe.optimized_mlir
    assert 'tessera.presburger_constraints' in recipe.optimized_mlir
    before = _optimize.cache_info().hits
    assert candidate.prepare(tessera_opt=tool(), system=system).digest == recipe.digest
    assert _optimize.cache_info().hits == before + 1
    with pytest.raises(ValueError, match='exactly'):
        recipe.rank_buckets([{'K': 4}])
    with pytest.raises(ValueError, match='positive integers'):
        recipe.rank_buckets([{'M': 2, 'K': True, 'N': 3}])


def test_loop_refuses_nonzero_initialization():
    def bad(a: tessera.Tensor['M', 'K', 'f32'], b: tessera.Tensor['K', 'N', 'f32']):
        c = np.ones((a.shape[0], b.shape[1]), dtype=a.dtype)
        for i in range(a.shape[0]):
            for j in range(b.shape[1]):
                for k in range(a.shape[1]):
                    c[i, j] += a[i, k] * b[k, j]
        return c
    with pytest.raises(ValueError, match='differs'):
        recognize_matmul_loop(bad)


def test_unresolved_recipe_fails_before_tool_lookup():
    def fn(a, b):
        return tessera.matmul(a, b)
    builder = GraphIRBuilder()
    builder.lower(fn, prefer_abstract_trace=False)
    with pytest.raises(ValueError, match='GRAPH_IR_UNRESOLVED_ELEMENT_TYPE'):
        prepare_recipe(builder.module(), tessera_opt='/does/not/exist')


def test_jit_rank_api():
    @tessera.jit
    def kernel(a: tessera.Tensor['M', 'K', 'f32'], b: tessera.Tensor['K', 'N', 'f32']):
        return tessera.matmul(a, b)
    ranks = kernel.rank_parametric_buckets([{'M': 2, 'K': 4, 'N': 3}], tessera_opt=tool())
    assert ranks[0].retained
    assert not ranks[0].promotion_eligible


def test_native_recipe_cse_happens_before_buckets():
    def duplicate(a: tessera.Tensor['M', 'K', 'f32'], b: tessera.Tensor['K', 'N', 'f32']):
        x = tessera.matmul(a, b)
        y = tessera.matmul(a, b)
        return tessera.add(x, y)
    builder = GraphIRBuilder()
    builder.lower(duplicate, prefer_abstract_trace=False)
    recipe = prepare_recipe(builder.module(), tessera_opt=tool())
    assert recipe.oracle_mlir.count('tessera.matmul') == 2
    assert recipe.optimized_mlir.count('tessera.matmul') == 1
    assert recipe.rank_buckets([{'M': 2, 'K': 4, 'N': 8}])[0].retained


@pytest.mark.parametrize('empty_module', [True, False])
def test_jit_rank_recovers_unmaterialized_ast_recipe(empty_module):
    from tessera.compiler.graph_ir import GraphIRModule

    @tessera.jit
    def kernel(a: tessera.Tensor['M', 'K', 'f32'], b: tessera.Tensor['K', 'N', 'f32']):
        return tessera.matmul(a, b)

    # Trace-defer/auto_batch store an empty module; tracer-only starts at None.
    kernel._legacy_graph_ir = GraphIRModule() if empty_module else None
    kernel.graph_ir = GraphIRModule()
    ranks = kernel.rank_parametric_buckets([{'M': 2, 'K': 4, 'N': 3}], tessera_opt=tool())
    assert len(kernel._legacy_graph_ir.functions) == 1
    assert ranks[0].retained
    assert not ranks[0].promotion_eligible
