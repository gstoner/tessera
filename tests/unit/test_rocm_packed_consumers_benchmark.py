import numpy as np

from benchmarks.rocm.benchmark_rocm_packed_consumers import (
    _build_consumer_hsaco,
    _compiled_packed_weight,
)
from tessera.stdlib.quant import unpack_int4


def test_compiled_packed_weight_matches_dequant_gemm_physical_abi(monkeypatch):
    logical = np.arange(-8, 8, dtype=np.int8).reshape(4, 4)
    scales = np.ones((2, 4), dtype=np.float32)

    def host_model(values, logical_elements, kind, np_module):
        assert kind == "pack"
        assert logical_elements == values.size
        flat = values.reshape(-1)
        lo = (flat[0::2] & 0xF).astype(np.uint8)
        hi = (flat[1::2] & 0xF).astype(np.uint8)
        return (lo | (hi << 4)).view(np.int8)

    monkeypatch.setattr(
        "benchmarks.rocm.benchmark_rocm_packed_consumers."
        "rt._rocm_int4_storage_convert",
        host_model,
    )
    packed = _compiled_packed_weight(logical, scales, 2)
    assert packed.codes.shape == (4, 2)
    assert packed.codes.nbytes * 2 == logical.nbytes
    np.testing.assert_array_equal(unpack_int4(packed.codes).T, logical)


def test_packed_consumer_builder_emits_backend_owned_directives(monkeypatch):
    observed = []

    def fake_build(pass_name, directive, cache, key):
        observed.append((pass_name, directive, key))
        return b"\x7fELF"

    monkeypatch.setattr(
        "benchmarks.rocm.benchmark_rocm_packed_consumers."
        "rt._build_rocm_elementwise_hsaco",
        fake_build,
    )
    monkeypatch.setattr(
        "benchmarks.rocm.benchmark_rocm_packed_consumers.rt._rocm_chip",
        lambda: "gfx1151",
    )
    from benchmarks.rocm import benchmark_rocm_packed_consumers as packet
    packet._consumer_hsaco_cache.clear()
    for kind in ("relu", "sparse_gather", "cache_append"):
        assert _build_consumer_hsaco(kind) == b"\x7fELF"
    assert {item[2][1] for item in observed} == {
        "relu", "sparse_gather", "cache_append"
    }
    assert all(item[0] == "generate-rocm-int4-pack-kernel"
               for item in observed)
