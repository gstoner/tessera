from scripts.record_dtype_codegen_inventory import record
from tessera.dtype import canonical_dtypes, planned_gated_dtypes


def test_inventory_keeps_missing_contracts_and_separates_math_modes():
    rows = {row['dtype']: row for row in record()['rows']}
    assert set(rows) == canonical_dtypes() | planned_gated_dtypes()
    assert 'tf32' not in rows
    assert any(c.get('math_mode') == 'tf32' for c in rows['fp32']['targets']['nvidia_sm120']['contracts'])
    assert all(row['targets']['apple_gpu']['coverage'] == 'no_layered_contract' for row in rows.values())
    assert rows['uint8']['vocabulary'] == 'planned_gated'
    assert rows['int8']['targets']['x86_zen5_avx512']['contracts'][0]['accumulator'] == 'int32'


def test_inventory_projects_gfx1201_exact_matmul_proof_per_input_storage():
    rows = {row['dtype']: row for row in record()['rows']}
    proved = {'fp16', 'bf16', 'fp8_e4m3', 'fp8_e5m2', 'int8', 'int4'}
    for dtype in proved:
        proof = rows[dtype]['targets']['rocm_gfx1201'][
            'exact_device_operations'
        ]['tessera.matmul']
        assert proof['executor_id'] == 'rocm_gfx1201_compiled'
        assert proof['proof_build'] == 'llvm23.1.1+rocm10.0+gfx1201'

    for dtype in {'fp32', 'int32', 'fp4_e2m1', 'mxfp4'}:
        assert 'exact_device_operations' not in rows[dtype]['targets']['rocm_gfx1201']
