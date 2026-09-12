import pytest
from benchmarks.apple_gpu.record_denormal_policy import source
from tessera.compiler.scheduled_matmul import find_tessera_opt, run_tessera_opt

pytestmark = pytest.mark.skipif(find_tessera_opt() is None, reason='native compiler required')


def lower(text):
    return run_tessera_opt(find_tessera_opt(),text,
        '--pass-pipeline=builtin.module(tessera-tile-buffer-reuse,tessera-tile-buffer-arena{emit-apple-msl=true})')


def test_gradual_and_ftz_are_distinct_consumed_native_policies():
    gradual=lower(source('gradual')); ftz=lower(source('flush_to_zero'))
    assert 'tessera_ieee_mul' in gradual and 'tessera_ieee_div' in gradual
    assert 'tessera_ftz(tessera_ieee_mul(tessera_ftz(' in ftz
    assert 'tessera.apple.denormal_mode = "gradual"' in gradual
    with pytest.raises(RuntimeError,match='unsupported Apple denormal policy'):
        lower(source('silently_relax'))


def test_policy_rejects_fast_math_and_unproven_transcendentals():
    changed=source('gradual').replace('arith.mulf %x, %y : f32','math.exp2 %x : f32')
    with pytest.raises(RuntimeError,match='no proven lowering'):
        lower(changed)
