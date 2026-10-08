"""Recorder guard distinguishes actual validation argv from shell source."""
import pytest
from benchmarks.rocm.benchmark_composed_scaled_vjp import _is_validation_argv

@pytest.mark.parametrize("argv,expected",[
    (["bash","-c","pytest -q tests; python benchmark.py"],False),
    (["python","-m","pytest","-q"],True),
    (["python","/tmp/venv/bin/pytest","-q"],True),
    (["/tmp/bin/graphify","update","."],True),
    (["/tmp/bin/graphify","query","pytest"],False),
    (["/tmp/bin/graphify","query","scaled_matmul"],False),
    (["python","benchmarks/rocm/benchmark_composed_scaled_vjp.py"],False),
])
def test_validation_guard_uses_actual_arguments(argv,expected):
    assert _is_validation_argv(argv)==expected
