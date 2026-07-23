from benchmarks.rocm.benchmark_rocm_int4_terminal import _packed_wmma_artifact


def test_packed_terminal_benchmark_uses_direct_compiled_consumer() -> None:
    metadata = _packed_wmma_artifact((33, 17, 31)).metadata
    assert metadata["wmma_dtype"] == "int4"
    assert metadata["wmma_inputs_packed"] is True
    assert metadata["logical_mnk"] == (33, 17, 31)
    assert metadata["compiler_path"] == "rocm_compiled"
