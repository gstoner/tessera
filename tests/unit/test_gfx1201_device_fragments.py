"""Explicit owning-device gate; no fallback or rc=2-to-skip inside the run."""
import os

import pytest


@pytest.mark.parametrize('dtype', ['f16', 'bf16', 'e4m3', 'e5m2', 'int8', 'int4'])
def test_fragment_recorder_source_matches_typed_fixture(dtype):
    from benchmarks.rocm.benchmark_rocm_arch_fragments import Case, _source
    from tests.unit.test_rocm_arch_fragment_compiler import _typed_source, FIXTURE
    expected = FIXTURE.read_text() if dtype == 'f16' else _typed_source(dtype)
    assert _source(Case('gfx1201', 'rdna4_wmma', dtype)) == expected


@pytest.mark.skipif(os.environ.get('TESSERA_GFX1201_DEVICE_PROOF') != '1',
                    reason='requires explicitly selected gfx1201 device')
def test_compiler_emitted_fragments_match_matrix_oracle(tmp_path):
    from benchmarks.rocm.record_gfx1201_fragments import record
    record(tmp_path / 'fragments.json')
