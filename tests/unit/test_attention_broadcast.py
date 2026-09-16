import os
from dataclasses import replace
import numpy as np
import pytest
import tessera
from tessera.compiler.loop_idioms import recognize_attention_loop
from tessera.compiler.scheduled_matmul import find_tessera_opt, run_tessera_opt
from tessera.compiler.raised_attention import lower_attention_bucket, bind_attention_bucket
from tests.unit.test_attention_loop_idiom import grouped_masked_bias

def broadcast_masked_bias(q: tessera.Tensor['B','HQ','Q',4,'f32'], k: tessera.Tensor['B','HK','K',4,'f32'], v: tessera.Tensor['B','HK','K','V','f32'], bias: tessera.Tensor[1,1,'Q','K','f32']):  # noqa: F821
    out = np.zeros((q.shape[0], q.shape[1], q.shape[2], v.shape[3]), dtype=q.dtype)
    for batch in range(q.shape[0]):
        for head in range(q.shape[1]):
            for query in range(q.shape[2]):
                scores = np.zeros((k.shape[2],), dtype=q.dtype)
                for key in range(k.shape[2]):
                    for feature in range(q.shape[3]):
                        scores[key] += q[batch, head, query, feature] * k[batch, head // (q.shape[1] // k.shape[1]), key, feature]
                scores = scores + bias[0, 0, query, :]
                scores[:max(query + max(k.shape[2] - q.shape[2], 0) - 2, 0)] = -np.inf
                scores[query + max(k.shape[2] - q.shape[2], 0) + 1:] = -np.inf
                weights = np.exp(scores - np.max(scores))
                weights = weights / np.sum(weights)
                for key in range(k.shape[2]):
                    for value in range(v.shape[3]):
                        out[batch, head, query, value] += weights[key] * v[batch, head // (q.shape[1] // k.shape[1]), key, value]
    return out


# A key-padding mask: one additive row shared by every batch, head and query.
# (No docstring: the recognizer compares the complete function body AST.)
def key_padding_masked_bias(q: tessera.Tensor['B','HQ','Q',4,'f32'], k: tessera.Tensor['B','HK','K',4,'f32'], v: tessera.Tensor['B','HK','K','V','f32'], bias: tessera.Tensor[1,1,1,'K','f32']):  # noqa: F821
    out = np.zeros((q.shape[0], q.shape[1], q.shape[2], v.shape[3]), dtype=q.dtype)
    for batch in range(q.shape[0]):
        for head in range(q.shape[1]):
            for query in range(q.shape[2]):
                scores = np.zeros((k.shape[2],), dtype=q.dtype)
                for key in range(k.shape[2]):
                    for feature in range(q.shape[3]):
                        scores[key] += q[batch, head, query, feature] * k[batch, head // (q.shape[1] // k.shape[1]), key, feature]
                scores = scores + bias[0, 0, 0, :]
                scores[:max(query + max(k.shape[2] - q.shape[2], 0) - 2, 0)] = -np.inf
                scores[query + max(k.shape[2] - q.shape[2], 0) + 1:] = -np.inf
                weights = np.exp(scores - np.max(scores))
                weights = weights / np.sum(weights)
                for key in range(k.shape[2]):
                    for value in range(v.shape[3]):
                        out[batch, head, query, value] += weights[key] * v[batch, head // (q.shape[1] // k.shape[1]), key, value]
    return out


# A per-(batch, head, query) additive scalar broadcast over every key.
def query_masked_bias(q: tessera.Tensor['B','HQ','Q',4,'f32'], k: tessera.Tensor['B','HK','K',4,'f32'], v: tessera.Tensor['B','HK','K','V','f32'], bias: tessera.Tensor['B','HQ','Q',1,'f32']):  # noqa: F821
    out = np.zeros((q.shape[0], q.shape[1], q.shape[2], v.shape[3]), dtype=q.dtype)
    for batch in range(q.shape[0]):
        for head in range(q.shape[1]):
            for query in range(q.shape[2]):
                scores = np.zeros((k.shape[2],), dtype=q.dtype)
                for key in range(k.shape[2]):
                    for feature in range(q.shape[3]):
                        scores[key] += q[batch, head, query, feature] * k[batch, head // (q.shape[1] // k.shape[1]), key, feature]
                scores = scores + bias[batch, head, query, 0]
                scores[:max(query + max(k.shape[2] - q.shape[2], 0) - 2, 0)] = -np.inf
                scores[query + max(k.shape[2] - q.shape[2], 0) + 1:] = -np.inf
                weights = np.exp(scores - np.max(scores))
                weights = weights / np.sum(weights)
                for key in range(k.shape[2]):
                    for value in range(v.shape[3]):
                        out[batch, head, query, value] += weights[key] * v[batch, head // (q.shape[1] // k.shape[1]), key, value]
    return out


SOURCES = {
    'batch_head': (broadcast_masked_bias, lambda q, k: (1, 1, q, k)),
    'key_padding': (key_padding_masked_bias, lambda q, k: (1, 1, 1, k)),
    'per_query': (query_masked_bias, lambda q, k: (2, 4, q, 1)),
}


@pytest.mark.parametrize('form', sorted(SOURCES))
@pytest.mark.parametrize('qsize,ksize', [(3,5),(5,3)])
def test_broadcast_masks_have_native_indexing_and_physical_guards(qsize,ksize,form):
    source, physical = SOURCES[form]
    bias_shape = physical(qsize, ksize)
    tool = find_tessera_opt()
    if tool is None:
        pytest.skip('native compiler required')
    recipe = recognize_attention_loop(source).prepare(tessera_opt=str(tool))
    bucket, = recipe.instantiate_buckets([dict(B=2,HQ=4,HK=2,Q=qsize,K=ksize,V=2)],tessera_opt=str(tool))
    artifact = lower_attention_bucket(recipe,bucket,compiler=tool)
    assert artifact.bias_shape == bias_shape
    stated = 'bias_shape = array<i64: ' + ', '.join(str(d) for d in bias_shape) + '>'
    assert stated in artifact.tile_ir
    tampered = artifact.schedule_ir.replace(stated, stated.replace('array<i64: ', 'array<i64: 7, ', 1).replace(', ' + str(bias_shape[0]) + ',', ',', 1))
    assert tampered != artifact.schedule_ir
    with pytest.raises(RuntimeError,match='altered'):
        run_tessera_opt(tool,tampered,'--tessera-schedule-to-tile')
    if os.environ.get('TESSERA_TEST_RAISED_ATTENTION') != '1':
        return
    binding = bind_attention_bucket(recipe,bucket,compiler=tool)
    rng = np.random.default_rng(746)
    values = [rng.uniform(-.5,.5,shape).astype(np.float32) for shape in
              [(2,4,qsize,4),(2,2,ksize,4),(2,2,ksize,2),bias_shape]]
    if bias_shape[3] != 1:
        values[-1][..., ksize // 2] = -np.inf  # a masked key column, broadcast wherever the axis is 1
    else:
        # A per-(batch, head, query) constant cancels out of an exact softmax, so a
        # random value could not tell a kernel that reads bias[b, h, q] from one
        # that ignores the operand or reads a neighbour. Give each row a
        # power-of-two magnitude from a per-row class instead: adding 2**e to the
        # f32 logits rounds them to that magnitude's ulp before the softmax, so
        # the output carries the rounding signature of exactly the row the kernel
        # read (f32 on both sides; q/k are quantised to 1/16 so every logit is
        # exact and no rounding boundary depends on accumulation order).
        values[0] = (np.round(values[0] * 16) / 16).astype(np.float32)
        values[1] = (np.round(values[1] * 16) / 16).astype(np.float32)
        rows = np.arange(2 * 4 * qsize).reshape(2, 4, qsize)
        values[-1] = np.ldexp(np.float32(1), 10 + rows % 14).astype(np.float32)[..., None]
    expected = grouped_masked_bias(*values[:3],np.ascontiguousarray(np.broadcast_to(values[-1],(2,4,qsize,ksize))))
    actual = binding(*values)
    np.testing.assert_allclose(actual,expected,rtol=1e-5,atol=1e-6)
    if bias_shape[3] == 1:
        # The signature must be visible: the same inputs without the bias give a
        # materially different output, so the kernel consumed bias[b, h, q].
        unbiased = binding(*values[:3], np.zeros(bias_shape, np.float32))
        assert np.abs(actual - unbiased).max() > 1e-3, 'per-query bias signature not observed'
    if directory := os.environ.get('TESSERA_BROADCAST_EVIDENCE'):
        import hashlib
        import json
        from pathlib import Path
        Path(directory).mkdir(parents=True,exist_ok=True)
        Path(directory,f'attention_{form}_{qsize}_{ksize}.json').write_text(json.dumps(dict(
            dims=artifact.dims,bias_shape=artifact.bias_shape,max_abs_error=float(np.max(np.abs(actual-expected))),
            compiler_sha256=hashlib.sha256(tool.read_bytes()).hexdigest(),
            runtime_sha256=hashlib.sha256(Path(os.environ["TESSERA_NVIDIA_PTX_LAUNCH_LIB"]).read_bytes()).hexdigest(),
            schedule_sha256=hashlib.sha256(artifact.schedule_ir.encode()).hexdigest(),
            image_digest=binding.package.image.image_digest,abi=binding.package.descriptor.abi_id,
            native_execution=True,promotion_eligible=False),indent=2)+'\n')
    if bias_shape[3] != 1:
        values[-1][...,0,:] = -np.inf  # every key of the first query row: an empty row on the broadcast view
        with pytest.raises(ValueError,match='fully masked'):
            binding(*values)


def test_bias_axes_outside_the_broadcast_contract_are_refused():
    def wrong(q: tessera.Tensor['B','HQ','Q',4,'f32'], k: tessera.Tensor['B','HK','K',4,'f32'], v: tessera.Tensor['B','HK','K','V','f32'], bias: tessera.Tensor[1,1,'K','Q','f32']):  # noqa: F821
        return broadcast_masked_bias(q, k, v, bias)
    with pytest.raises(ValueError, match='attention extent or 1'):
        recognize_attention_loop(wrong)
