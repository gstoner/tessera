"""Physical ingest policies are semantic and may not be silently normalized."""
import subprocess

import pytest

from tessera.compiler.scheduled_matmul import find_tessera_opt
from tests.device.rocm.test_native_nvfp4_ingest_leaf import directive


@pytest.fixture(scope="module")
def compiler():
    tool = find_tessera_opt()
    if tool is None:
        pytest.skip("matching compiler required")
    probe = subprocess.run([str(tool), "--generate-rocm-fpquant-kernel"],
        input=directive(7,64,(0,3,7)),text=True,capture_output=True)
    if probe.returncode and "Unknown command line argument" in probe.stderr:
        pytest.skip("ROCm compiler required")
    assert probe.returncode == 0, probe.stderr
    assert "gpu.func" in probe.stdout
    assert "tessera_rocm.nvfp4_requantize" not in probe.stdout
    return tool


@pytest.mark.parametrize("old,new", [
    ('execution_mode = "explicit_scale_requantization"', 'execution_mode = "approximate"'),
    ('source_layout = "e2m1_row_k_e4m3_k16_projection_global"', 'source_layout = "e2m1"'),
    ('destination_layout = "e2m1_row_k_e8m0_k32_group_n"', 'destination_layout = "transposed"'),
    ('arch = "gfx1201",', 'arch = "gfx1151",'),
    ('k = 64 : i64', 'k = 63 : i64'),
    ('n = 7 : i64', 'n = 8 : i64'),
    ('[0 : i64, 3 : i64, 7 : i64]', '[0 : i64, 0 : i64, 7 : i64]'),
    ('[0 : i64, 3 : i64, 7 : i64]', '[1 : i64, 3 : i64, 7 : i64]'),
    ('n = 7 : i64', 'n = 9223372036854775807 : i64'),
])
def test_target_rejects_conflicting_ingest_contract(compiler,old,new):
    text = directive(7,64,(0,3,7))
    assert old in text
    result = subprocess.run([str(compiler), "--generate-rocm-fpquant-kernel"],
        input=text.replace(old,new),text=True,capture_output=True)
    assert result.returncode != 0
    assert "requires" in result.stderr


def test_materializer_rejects_wrong_module_architecture(compiler):
    text = directive(7,64,(0,3,7)).replace(
        'tessera.arch = "gfx1201"', 'tessera.arch = "gfx1151"')
    result = subprocess.run([str(compiler), "--generate-rocm-fpquant-kernel"],
        input=text,text=True,capture_output=True)
    assert result.returncode != 0
    assert "requires an exact gfx1201 module" in result.stderr
