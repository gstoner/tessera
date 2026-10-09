"""Owning-device interleaving of pinned page staging and native typed math."""
import os
from unittest.mock import patch

import numpy as np
import pytest
import tessera as ts
from tessera import runtime as rt
from tests.unit.test_public_movement_frontend import paged
from tests.unit.test_strided_paged_kv_runtime import pages
from tests.device.rocm.test_native_math_package_jit import FUNCTIONS
from tests.device.rocm.test_native_math_widening import FUNCTIONS as WIDENING


@pytest.mark.skipif(os.environ.get('TESSERA_ROCM_MOVEMENT_DEVICE_PROOF') != '1',
                    reason='requires explicit owning ROCm device proof')
@pytest.mark.parametrize('storage', ['f32', 'f16', 'bf16'])
@pytest.mark.parametrize('kind', ['sqrt', 'add'])
def test_pinned_pages_and_typed_math_share_capacity_without_input_content(storage, kind, monkeypatch):
    arch = os.environ['TESSERA_ROCM_CHIP']
    assert arch in {'gfx1151', 'gfx1201'} and rt._rocm_live_arch() == arch
    monkeypatch.delenv('TESSERA_ROCM_MOVEMENT_PINNED_STAGING', raising=False)
    monkeypatch.setenv('TESSERA_ROCM_NATIVE_MATH', '1')
    monkeypatch.setenv('TESSERA_ROCM_MATH_STAGING_REUSE', '1')
    dtype = {'f32': np.float32, 'f16': np.float16,
             'bf16': pytest.importorskip('ml_dtypes').bfloat16}[storage]
    x = pages('padded')
    table = np.array([2, 0, 3, 1], np.int32)
    gather = ts.jit(target='rocm_'+arch, native_required=True)(paged)
    math = ts.jit(target='rocm_'+arch, native_required=True)((FUNCTIONS if storage == 'f32' else WIDENING)[kind])
    values = [np.full((7, 13), .5, dtype)]
    if kind == 'add':
        values.append(np.full((7, 13), .75, dtype))
    np.testing.assert_array_equal(gather(x, table), x[table].reshape(-1, 3, 8)[1:6])
    math(*values)
    assert math.execution_kind == 'native_gpu'
    assert math.runtime_artifact().launch_descriptor.provenance['native_math']['kind'] == kind

    def forbidden(*args, **kwargs):
        raise AssertionError('compiler invocation during warm interleaving')

    with patch('subprocess.run', forbidden), patch('subprocess.Popen', forbidden), patch('subprocess.check_output', forbidden):
        for iteration in range(4):
            x[...] *= np.float32(-.75)
            table[:] = table[::-1]
            page_oracle = x[table].reshape(-1, 3, 8)[1:6]
            retained = gather(x, table)
            np.testing.assert_array_equal(retained, page_oracle)
            for ordinal, value in enumerate(values):
                value[...] = .25 + .0625*iteration + .5*ordinal
            oracle = np.sqrt(values[0].astype(np.float32)) if kind == 'sqrt' else values[0].astype(np.float32)+values[1].astype(np.float32)
            np.testing.assert_allclose(math(*values), oracle, rtol=2e-5, atol=2e-5)
            np.testing.assert_array_equal(gather(x, table), page_oracle)
            np.testing.assert_array_equal(retained, page_oracle)
            rt._clear_rocm_native_image_cache()
            np.testing.assert_array_equal(gather(x, table), page_oracle)
            np.testing.assert_allclose(math(*values), oracle, rtol=2e-5, atol=2e-5)
    rt._clear_rocm_native_image_cache()
