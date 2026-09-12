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


@pytest.mark.parametrize('qsize,ksize', [(3,5),(5,3)])
def test_broadcast_masks_have_native_indexing_and_physical_guards(qsize,ksize):
    tool = find_tessera_opt()
    if tool is None:
        pytest.skip('native compiler required')
    recipe = recognize_attention_loop(broadcast_masked_bias).prepare(tessera_opt=str(tool))
    bucket, = recipe.instantiate_buckets([dict(B=2,HQ=4,HK=2,Q=qsize,K=ksize,V=2)],tessera_opt=str(tool))
    artifact = lower_attention_bucket(recipe,bucket,compiler=tool)
    assert artifact.bias_shape == (1,1,qsize,ksize)
    assert f'bias_shape = array<i64: 1, 1, {qsize}, {ksize}>' in artifact.tile_ir
    tampered = artifact.schedule_ir.replace('bias_shape = array<i64: 1, 1,', 'bias_shape = array<i64: 2, 1,')
    with pytest.raises(RuntimeError,match='altered'):
        run_tessera_opt(tool,tampered,'--tessera-schedule-to-tile')
    if os.environ.get('TESSERA_TEST_RAISED_ATTENTION') != '1':
        return
    binding = bind_attention_bucket(recipe,bucket,compiler=tool)
    rng = np.random.default_rng(746)
    values = [rng.uniform(-.5,.5,shape).astype(np.float32) for shape in
              [(2,4,qsize,4),(2,2,ksize,4),(2,2,ksize,2),(1,1,qsize,ksize)]]
    values[-1][..., ksize // 2] = -np.inf
    expected = grouped_masked_bias(*values[:3],np.broadcast_to(values[-1],(2,4,qsize,ksize)))
    actual = binding(*values)
    np.testing.assert_allclose(actual,expected,rtol=1e-5,atol=1e-6)
    if directory := os.environ.get('TESSERA_BROADCAST_EVIDENCE'):
        import hashlib
        import json
        from pathlib import Path
        Path(directory).mkdir(parents=True,exist_ok=True)
        Path(directory,f'attention_{qsize}_{ksize}.json').write_text(json.dumps(dict(
            dims=artifact.dims,bias_shape=artifact.bias_shape,max_abs_error=float(np.max(np.abs(actual-expected))),
            compiler_sha256=hashlib.sha256(tool.read_bytes()).hexdigest(),
            runtime_sha256=hashlib.sha256(Path(os.environ["TESSERA_NVIDIA_PTX_LAUNCH_LIB"]).read_bytes()).hexdigest(),
            schedule_sha256=hashlib.sha256(artifact.schedule_ir.encode()).hexdigest(),
            image_digest=binding.package.image.image_digest,abi=binding.package.descriptor.abi_id,
            native_execution=True,promotion_eligible=False),indent=2)+'\n')
    values[-1][...,0,:] = -np.inf
    with pytest.raises(ValueError,match='fully masked'):
        binding(*values)
