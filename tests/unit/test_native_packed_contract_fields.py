"""Malformed native metadata fails before descriptor construction or device use."""
import pytest
from tessera.compiler import rocm_mxfp4_storage_native as storage
from tessera.compiler import rocm_nvfp4_ingest_native as ingest


@pytest.mark.parametrize("module,tile_op,target_op", [
    (storage, "tile.mxfp4_folded_storage_kernel", "tessera_rocm.mxfp4_folded_storage"),
    (ingest, "tile.nvfp4_requantize_kernel", "tessera_rocm.nvfp4_requantize"),
])
@pytest.mark.parametrize("missing", ["bindings", "n", "k", "schedule_hash", "name"])
def test_missing_native_field_is_a_contract_error(module, tile_op, target_op, missing):
    tile = tile_op + ' bindings = ["a", "b", "c", "d", "e", "f"]'
    fields = {
        "n": "n = 16 : i64",
        "k": "k = 64 : i64",
        "schedule_hash": 'schedule_hash = "' + "a" * 64 + '"',
        "name": 'name = "entry"',
        "row_offsets": "row_offsets = [0, 16]",
        "storage_contract": 'storage_contract = "' + storage.MXFP4_STORAGE_CONTRACT + '"',
    }
    if missing == "bindings":
        tile = tile_op
    target = target_op + " " + " ".join(value for key, value in fields.items() if key != missing)
    with pytest.raises(ValueError, match="native contract field is missing"):
        module._contract(tile, target)
