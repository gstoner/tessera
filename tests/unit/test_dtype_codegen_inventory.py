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
