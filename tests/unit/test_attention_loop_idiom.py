import inspect
import os
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


def test_raised_attention_bucket_projects_native_descriptor_and_replays_parent():
    from dataclasses import replace
    from tessera.compiler.raised_attention import lower_attention_bucket
    tool = find_tessera_opt()
    if tool is None:
        pytest.skip('native compiler required')
    recipe = recognize_attention_loop(attention).prepare(tessera_opt=str(tool))
    instance = recipe.instantiate_buckets([dict(B=1,H=2,Q=3,K=5,V=2)],tessera_opt=str(tool))[0]
    artifact = lower_attention_bucket(recipe,instance,compiler=tool)
    assert artifact.dims == (1,2,2,3,5,4,2)
    assert artifact.scale == 1.0 and artifact.output_name == 'out'
    with pytest.raises(ValueError,match='replay'):
        lower_attention_bucket(recipe,replace(instance,mlir=instance.mlir.replace('scale = 1.', 'scale = 2.')),compiler=tool)


@pytest.mark.skipif(os.environ.get('TESSERA_TEST_RAISED_ATTENTION') != '1', reason='owning CUDA device required')
@pytest.mark.parametrize('scale',[1.,.5])
def test_raised_attention_executes_both_buckets_without_graph_reconstruction(monkeypatch,scale):
    from tessera.compiler.raised_attention import bind_attention_bucket
    from tessera.compiler.graph_ir import GraphIRModule
    tool = find_tessera_opt()
    if scale != 1.:
        lines,line = inspect.getsourcelines(attention)
        source = textwrap.dedent(''.join(lines)).replace('                weights = np.exp',
            '                scores = scores * 0.5\n                weights = np.exp')
        monkeypatch.setattr(inspect,'getsourcelines',lambda fn:(source.splitlines(keepends=True),line))
    recipe = recognize_attention_loop(attention).prepare(tessera_opt=str(tool))
    instances = recipe.instantiate_buckets([dict(B=1,H=2,Q=3,K=5,V=2),dict(B=1,H=2,Q=5,K=7,V=2)],tessera_opt=str(tool))
    def forbidden(*args,**kwargs):
        pytest.fail('Graph reconstruction during native bucket binding')
    monkeypatch.setattr(GraphIRModule,'__init__',forbidden)
    for instance in instances:
        bucket = dict(instance.bindings)
        bound = bind_attention_bucket(recipe,instance,compiler=tool)
        rng = np.random.default_rng(9)
        q,k,v = [rng.uniform(-.5,.5,shape).astype(np.float32) for shape in
                 [(1,2,bucket['Q'],4),(1,2,bucket['K'],4),(1,2,bucket['K'],2)]]
        np.testing.assert_allclose(bound(q,k,v),attention(q*scale,k,v),rtol=1e-5,atol=1e-6)


@pytest.mark.parametrize('target', ['x86','apple_gpu'])
def test_raised_attention_replays_target_specific_parent(target):
    from tessera.compiler.raised_attention import lower_attention_bucket
    from tessera.compiler.native_attention_contract import verify_attention_ancestry
    tool = find_tessera_opt()
    if tool is None:
        pytest.skip('native compiler required')
    recipe = recognize_attention_loop(attention).prepare(tessera_opt=str(tool))
    instance = recipe.instantiate_buckets([dict(B=1,H=2,Q=3,K=5,V=4)],tessera_opt=str(tool))[0]
    artifact = lower_attention_bucket(recipe,instance,compiler=tool,target=target)
    verify_attention_ancestry(artifact,target=target,architecture=artifact.architecture)
    assert artifact.dims == (1,2,2,3,5,4,4)
    assert artifact.backward_lse_selection == ('recompute' if target == 'apple_gpu' else 'saved')
    with pytest.raises(ValueError,match='no admitted backend'):
        lower_attention_bucket(recipe,instance,compiler=tool,target='rocm_gfx1151')


@pytest.mark.skipif(os.environ.get('TESSERA_TEST_RAISED_ATTENTION_X86') != '1', reason='owning Zen 5 host required')
def test_raised_attention_executes_x86_without_graph_reconstruction(monkeypatch):
    from tessera.compiler.raised_attention import bind_attention_bucket
    from tessera.compiler.graph_ir import GraphIRModule
    tool = find_tessera_opt()
    recipe = recognize_attention_loop(attention).prepare(tessera_opt=str(tool))
    instance = recipe.instantiate_buckets([dict(B=1,H=2,Q=3,K=5,V=4)],tessera_opt=str(tool))[0]
    def forbidden(*args,**kwargs):
        pytest.fail('Graph reconstruction during native x86 binding')
    monkeypatch.setattr(GraphIRModule,'__init__',forbidden)
    bound = bind_attention_bucket(recipe,instance,compiler=tool,target='x86')
    rng = np.random.default_rng(912)
    q,k,v = [rng.uniform(-.5,.5,shape).astype(np.float32) for shape in [(1,2,3,4),(1,2,5,4),(1,2,5,4)]]
    np.testing.assert_allclose(bound(q,k,v),attention(q,k,v),rtol=1e-5,atol=1e-6)


@pytest.mark.parametrize('literal,accepted',[('0.5',True),('0.1',False),('True',False),('0.0',False)])
def test_attention_literal_scale_recognition(monkeypatch,literal,accepted):
    lines,line = inspect.getsourcelines(attention)
    source = textwrap.dedent(''.join(lines)).replace('                weights = np.exp',
        '                scores = scores * '+literal+'\n                weights = np.exp')
    monkeypatch.setattr(inspect,'getsourcelines',lambda fn:(source.splitlines(keepends=True),line))
    if not accepted:
        with pytest.raises(ValueError,match='scale'):
            recognize_attention_loop(attention)
    else:
        candidate = recognize_attention_loop(attention)
        assert candidate.raised_module.functions[0].body[0].kwargs['scale'] == .5
        assert not candidate.promotion_eligible


def grouped_attention(q: tessera.Tensor['B','HQ','Q',4,'f32'], k: tessera.Tensor['B','HK','K',4,'f32'], v: tessera.Tensor['B','HK','K','V','f32']):  # noqa: F821 -- symbolic Tensor annotation strings
    out = np.zeros((q.shape[0], q.shape[1], q.shape[2], v.shape[3]), dtype=q.dtype)
    for batch in range(q.shape[0]):
        for head in range(q.shape[1]):
            for query in range(q.shape[2]):
                scores = np.zeros((k.shape[2],), dtype=q.dtype)
                for key in range(k.shape[2]):
                    for feature in range(q.shape[3]):
                        scores[key] += q[batch, head, query, feature] * k[batch, head // (q.shape[1] // k.shape[1]), key, feature]
                scores[query + max(k.shape[2] - q.shape[2], 0) + 1:] = -np.inf
                weights = np.exp(scores - np.max(scores))
                weights = weights / np.sum(weights)
                for key in range(k.shape[2]):
                    for value in range(v.shape[3]):
                        out[batch, head, query, value] += weights[key] * v[batch, head // (q.shape[1] // k.shape[1]), key, value]
    return out


def test_grouped_causal_attention_recognition_and_native_policy():
    from tessera.compiler.raised_attention import lower_attention_bucket
    candidate = recognize_attention_loop(grouped_attention)
    assert candidate.raised_module.functions[0].body[0].kwargs['causal'] is True
    tool = find_tessera_opt()
    if tool is None:
        pytest.skip('native compiler required')
    recipe = candidate.prepare(tessera_opt=str(tool))
    instance = recipe.instantiate_buckets([dict(B=1,HQ=4,HK=2,Q=3,K=5,V=2)],tessera_opt=str(tool))[0]
    artifact = lower_attention_bucket(recipe,instance,compiler=tool)
    assert artifact.dims == (1,4,2,3,5,4,2) and artifact.causal


@pytest.mark.skipif(os.environ.get('TESSERA_TEST_RAISED_ATTENTION') != '1', reason='owning CUDA device required')
@pytest.mark.parametrize('qsize,ksize',[(3,5),(5,3)])
def test_grouped_ragged_causal_attention_executes(qsize,ksize):
    from tessera.compiler.raised_attention import bind_attention_bucket
    tool = find_tessera_opt()
    recipe = recognize_attention_loop(grouped_attention).prepare(tessera_opt=str(tool))
    instance = recipe.instantiate_buckets([dict(B=1,HQ=4,HK=2,Q=qsize,K=ksize,V=2)],tessera_opt=str(tool))[0]
    bound = bind_attention_bucket(recipe,instance,compiler=tool)
    rng = np.random.default_rng(742)
    q,k,v = [rng.uniform(-.5,.5,shape).astype(np.float32) for shape in [(1,4,qsize,4),(1,2,ksize,4),(1,2,ksize,2)]]
    np.testing.assert_allclose(bound(q,k,v),grouped_attention(q,k,v),rtol=1e-5,atol=1e-6)


def test_grouped_attention_refuses_nondivisible_native_bucket():
    import subprocess
    tool = find_tessera_opt()
    if tool is None:
        pytest.skip('native compiler required')
    recipe = recognize_attention_loop(grouped_attention).prepare(tessera_opt=str(tool))
    with pytest.raises(subprocess.CalledProcessError) as failure:
        recipe.instantiate_buckets([dict(B=1,HQ=3,HK=2,Q=3,K=5,V=2)],tessera_opt=str(tool))
    assert 'positive multiple of key heads' in failure.value.stderr


def test_attention_refuses_different_causal_alignment(monkeypatch):
    import inspect
    original = inspect.getsourcelines
    def source(fn):
        lines,first = original(fn)
        return [line.replace('query + max(k.shape[2] - q.shape[2], 0) + 1:', 'query + 1:') for line in lines],first
    monkeypatch.setattr(inspect,'getsourcelines',source)
    with pytest.raises(ValueError,match='complete'):
        recognize_attention_loop(grouped_attention)


def window_attention(q: tessera.Tensor['B','HQ','Q',4,'f32'], k: tessera.Tensor['B','HK','K',4,'f32'], v: tessera.Tensor['B','HK','K','V','f32']):  # noqa: F821 -- symbolic Tensor annotation strings
    out = np.zeros((q.shape[0], q.shape[1], q.shape[2], v.shape[3]), dtype=q.dtype)
    for batch in range(q.shape[0]):
        for head in range(q.shape[1]):
            for query in range(q.shape[2]):
                scores = np.zeros((k.shape[2],), dtype=q.dtype)
                for key in range(k.shape[2]):
                    for feature in range(q.shape[3]):
                        scores[key] += q[batch, head, query, feature] * k[batch, head // (q.shape[1] // k.shape[1]), key, feature]
                scores[:max(query + max(k.shape[2] - q.shape[2], 0) - 2, 0)] = -np.inf
                scores[query + max(k.shape[2] - q.shape[2], 0) + 1 + 1:] = -np.inf
                weights = np.exp(scores - np.max(scores))
                weights = weights / np.sum(weights)
                for key in range(k.shape[2]):
                    for value in range(v.shape[3]):
                        out[batch, head, query, value] += weights[key] * v[batch, head // (q.shape[1] // k.shape[1]), key, value]
    return out



@pytest.mark.skipif(os.environ.get('TESSERA_TEST_RAISED_ATTENTION') != '1',reason='owning CUDA host')
def test_windowed_gqa_executes():
    from tessera.compiler.raised_attention import bind_attention_bucket
    tool = find_tessera_opt()
    if tool is None: pytest.skip('native compiler')
    candidate = recognize_attention_loop(window_attention)
    recipe = candidate.prepare(tessera_opt=str(tool))
    instance = recipe.instantiate_buckets([dict(B=1,HQ=4,HK=2,Q=3,K=5,V=2)],tessera_opt=str(tool))[0]
    binding = bind_attention_bucket(recipe,instance,compiler=tool)
    assert (binding.artifact.window_left,binding.artifact.window_right) == (2,1)
    rng = np.random.default_rng(744)
    q,k,v = [rng.normal(size=s).astype(np.float32) for s in [(1,4,3,4),(1,2,5,4),(1,2,5,2)]]
    np.testing.assert_allclose(binding(q,k,v),window_attention(q,k,v),rtol=1e-5,atol=1e-6)


def test_window_attention_empty_rows_pruned_before_native_execution():
    tool = find_tessera_opt()
    if tool is None: pytest.skip('native compiler')
    recipe = recognize_attention_loop(window_attention).prepare(tessera_opt=str(tool))
    with pytest.raises(ValueError,match='rejected'):
        recipe.instantiate_buckets([dict(B=1,HQ=4,HK=2,Q=8,K=2,V=2)],tessera_opt=str(tool))
    assert 'presburger_constraints' in recipe.optimized_mlir


def test_caller_constraints_cannot_drop_window_legality():
    from tessera.compiler.presburger import PresburgerSystem, PresburgerConstraint
    tool = find_tessera_opt()
    if tool is None: pytest.skip('native compiler')
    candidate = recognize_attention_loop(window_attention)
    permissive = PresburgerSystem(('Q',),(PresburgerConstraint('ge',(1,),0),))
    recipe = candidate.prepare(tessera_opt=str(tool),system=permissive)
    assert not recipe.rank_buckets([dict(B=1,HQ=4,HK=2,Q=8,K=2,V=2)])[0].retained


def attention_bias(q: tessera.Tensor['B','H','Q',4,'f32'], k: tessera.Tensor['B','H','K',4,'f32'], v: tessera.Tensor['B','H','K','V','f32'], bias: tessera.Tensor['B','H','Q','K','f32']):  # noqa: F821 -- symbolic Tensor annotation strings
    out = np.zeros((q.shape[0], q.shape[1], q.shape[2], v.shape[3]), dtype=q.dtype)
    for batch in range(q.shape[0]):
        for head in range(q.shape[1]):
            for query in range(q.shape[2]):
                scores = np.zeros((k.shape[2],), dtype=q.dtype)
                for key in range(k.shape[2]):
                    for feature in range(q.shape[3]):
                        scores[key] += q[batch, head, query, feature] * k[batch, head, key, feature]
                scores = scores + bias[batch, head, query, :]
                weights = np.exp(scores - np.max(scores))
                weights = weights / np.sum(weights)
                for key in range(k.shape[2]):
                    for value in range(v.shape[3]):
                        out[batch, head, query, value] += weights[key] * v[batch, head, key, value]
    return out


def test_full_shape_additive_bias_native_binding():
    tool = find_tessera_opt()
    if tool is None:
        pytest.skip('native compiler required')
    recipe = recognize_attention_loop(attention_bias).prepare(tessera_opt=str(tool))
    instance, = recipe.instantiate_buckets([{'B':1,'H':2,'Q':3,'K':5,'V':2}],tessera_opt=str(tool))
    from tessera.compiler.raised_attention import lower_attention_bucket, bind_attention_bucket
    artifact = lower_attention_bucket(recipe,instance,compiler=tool)
    assert artifact.bias_name == 'bias'
    if os.environ.get('TESSERA_TEST_RAISED_ATTENTION') != '1':
        pytest.skip('owning NVIDIA device required')
    binding = bind_attention_bucket(recipe,instance,compiler=tool)
    rng = np.random.default_rng(873)
    values = [rng.uniform(-.5,.5,shape).astype(np.float32) for shape in [(1,2,3,4),(1,2,5,4),(1,2,5,2),(1,2,3,5)]]
    np.testing.assert_allclose(binding(*values),attention_bias(*values),rtol=1e-5,atol=1e-6)
    values[-1][..., ::2] = -np.inf
    np.testing.assert_allclose(binding(*values), attention_bias(*values), rtol=1e-5, atol=1e-6)
    values[-1][0, 0, 0, :] = -np.inf
    with pytest.raises(ValueError, match='fully masked'):
        binding(*values)
    values[-1][0,0,0,0] = np.nan
    with pytest.raises(ValueError,match='finite'):
        binding(*values)


def test_additive_mask_composes_with_causal_window_and_refuses_empty_rows():
    from types import SimpleNamespace
    from tessera.compiler.raised_attention import validate_mask_rows
    artifact = SimpleNamespace(dims=(1, 2, 1, 3, 5, 4, 4), causal=True, window_left=1, window_right=-1)
    bias = np.zeros((1, 2, 3, 5), np.float32)
    bias[..., 0] = -np.inf
    validate_mask_rows(artifact, bias)
    # First query can read keys 1 and 2 only; finite keys outside that range
    # must not make an otherwise fully masked row admissible.
    bias[..., 0, 1:3] = -np.inf
    with pytest.raises(ValueError, match='fully masked'):
        validate_mask_rows(artifact, bias)


def grouped_masked_bias(q: tessera.Tensor['B','HQ','Q',4,'f32'], k: tessera.Tensor['B','HK','K',4,'f32'], v: tessera.Tensor['B','HK','K','V','f32'], bias: tessera.Tensor['B','HQ','Q','K','f32']):  # noqa: F821
    out = np.zeros((q.shape[0], q.shape[1], q.shape[2], v.shape[3]), dtype=q.dtype)
    for batch in range(q.shape[0]):
        for head in range(q.shape[1]):
            for query in range(q.shape[2]):
                scores = np.zeros((k.shape[2],), dtype=q.dtype)
                for key in range(k.shape[2]):
                    for feature in range(q.shape[3]):
                        scores[key] += q[batch, head, query, feature] * k[batch, head // (q.shape[1] // k.shape[1]), key, feature]
                scores = scores + bias[batch, head, query, :]
                scores[:max(query + max(k.shape[2] - q.shape[2], 0) - 2, 0)] = -np.inf
                scores[query + max(k.shape[2] - q.shape[2], 0) + 1:] = -np.inf
                weights = np.exp(scores - np.max(scores))
                weights = weights / np.sum(weights)
                for key in range(k.shape[2]):
                    for value in range(v.shape[3]):
                        out[batch, head, query, value] += weights[key] * v[batch, head // (q.shape[1] // k.shape[1]), key, value]
    return out


@pytest.mark.skipif(os.environ.get('TESSERA_TEST_RAISED_ATTENTION') != '1', reason='owning CUDA device required')
@pytest.mark.parametrize('qsize,ksize', [(3, 5), (5, 3)])
def test_irregular_masks_compose_with_ragged_gqa_causal_windows(qsize, ksize):
    from tessera.compiler.raised_attention import bind_attention_bucket
    tool = find_tessera_opt()
    recipe = recognize_attention_loop(grouped_masked_bias).prepare(tessera_opt=str(tool))
    bucket, = recipe.instantiate_buckets([dict(B=1,HQ=4,HK=2,Q=qsize,K=ksize,V=2)],tessera_opt=str(tool))
    binding = bind_attention_bucket(recipe, bucket, compiler=tool)
    rng = np.random.default_rng(745)
    values = [rng.uniform(-.5,.5,shape).astype(np.float32) for shape in
              [(1,4,qsize,4),(1,2,ksize,4),(1,2,ksize,2),(1,4,qsize,ksize)]]
    values[-1][..., ksize // 2] = -np.inf
    np.testing.assert_allclose(binding(*values), grouped_masked_bias(*values), rtol=1e-5, atol=1e-6)
    center = max(ksize-qsize, 0)
    values[-1][0,0,0,max(center-2,0):center+1] = -np.inf
    with pytest.raises(ValueError, match='fully masked'):
        binding(*values)
