import inspect
import textwrap
import numpy as np
import pytest
import tessera
from tessera.compiler.loop_idioms import recognize_attention_loop
from tessera.compiler.scheduled_matmul import find_tessera_opt, run_tessera_opt


def attention(q: tessera.Tensor['B','H','Q',4,'f32'], k: tessera.Tensor['B','H','K',4,'f32'], v: tessera.Tensor['B','H','K','V','f32']):  # noqa: F821 -- symbolic Tensor annotation strings
    out = np.zeros((q.shape[0], q.shape[1], q.shape[2], v.shape[3]), dtype=q.dtype)
    for batch in range(q.shape[0]):
        for head in range(q.shape[1]):
            for query in range(q.shape[2]):
                scores = np.zeros((k.shape[2],), dtype=q.dtype)
                for key in range(k.shape[2]):
                    for feature in range(q.shape[3]):
                        scores[key] += q[batch, head, query, feature] * k[batch, head, key, feature]
                weights = np.exp(scores - np.max(scores))
                weights = weights / np.sum(weights)
                for key in range(k.shape[2]):
                    for value in range(v.shape[3]):
                        out[batch, head, query, value] += weights[key] * v[batch, head, key, value]
    return out


def test_attention_candidate_has_native_schedule_path_and_retains_oracle():
    candidate = recognize_attention_loop(attention)
    assert not candidate.promotion_eligible
    assert 'scores[key] +=' in candidate.source_oracle
    rng = np.random.default_rng(77)
    q,k,v = [rng.uniform(-.5,.5,shape).astype(np.float32) for shape in [(1,2,3,4),(1,2,5,4),(1,2,5,2)]]
    scores = q @ k.swapaxes(-1,-2)
    weights = np.exp(scores-scores.max(axis=-1,keepdims=True))
    expected = weights/weights.sum(axis=-1,keepdims=True) @ v
    np.testing.assert_allclose(attention(q,k,v),expected,rtol=1e-5,atol=1e-6)
    if find_tessera_opt() is None:
        pytest.skip('native Schedule compiler required')
    tool = str(find_tessera_opt())
    recipe = candidate.prepare(tessera_opt=tool)
    instances = recipe.instantiate_buckets([
        {'B':1,'H':2,'Q':3,'K':5,'V':2},
        {'B':1,'H':2,'Q':5,'K':7,'V':2}],tessera_opt=tool)
    assert len({instance.recipe_digest for instance in instances}) == 1
    for instance in instances:
        assert 'tensor<?' not in instance.mlir
        targeted = instance.mlir.replace('module attributes {','module attributes {tessera.target = "nvidia_sm120", tessera.arch = "sm_120", tessera.launch_bindings = ["q", "k", "v", "out"], ',1)
        scheduled = run_tessera_opt(find_tessera_opt(),targeted,'--tessera-graph-to-schedule')
        lowered = run_tessera_opt(find_tessera_opt(),scheduled,'--tessera-schedule-to-tile')
        assert 'tile.attention_kernel' in lowered



@pytest.mark.parametrize('before,after',[
    ('np.max(scores)','np.min(scores)'),
    ('weights / np.sum(weights)','weights * np.sum(weights)'),
    ('np.zeros','np.ones'),
    ('return out','print(out)\n    return out'),
])
def test_attention_recognition_refuses_changed_math_or_effects(monkeypatch,before,after):
    lines,line = inspect.getsourcelines(attention)
    source = textwrap.dedent(''.join(lines)).replace(before,after)
    monkeypatch.setattr(inspect,'getsourcelines',lambda fn:(source.splitlines(keepends=True),line))
    with pytest.raises(ValueError,match='recurrence'):
        recognize_attention_loop(attention)


def test_attention_instantiation_refuses_stale_head_width():
    tool = find_tessera_opt()
    if tool is None:
        pytest.skip('native Schedule compiler required')
    recipe = recognize_attention_loop(attention).prepare(tessera_opt=str(tool))
    assert 'head_dim = 4' in recipe.optimized_mlir
    with pytest.raises(RuntimeError,match='contradicts'):
        run_tessera_opt(tool,recipe.optimized_mlir.replace('head_dim = 4','head_dim = 8'),
                       '--tessera-symdim-equality=instantiate=B:1;H:2;Q:3;K:5;V:2')
