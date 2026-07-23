from benchmarks.rocm.benchmark_rocm_lds_arena_occupancy import (
    _arena_module,
    _dynamic_arena_module,
)


def test_lds_arena_benchmark_emits_executable_address_space_3_allocation():
    source = _arena_module(16384)
    assert "memref<16384xi8, 3>" in source
    assert 'memref.global "private" @arena_storage_16384' in source
    assert "memref.get_global @arena_storage_16384" in source
    assert "memref.store" in source
    assert "gpu.func @arena_16384(%out: memref<?xi8>) kernel" in source
    assert "gpu.barrier" in source


def test_lds_arena_benchmark_emits_runtime_sized_launch_allocation():
    source = _dynamic_arena_module()
    assert "memref.alloca(%n) : memref<?xi8, 3>" in source
    assert "gpu.func @arena_dynamic" in source
