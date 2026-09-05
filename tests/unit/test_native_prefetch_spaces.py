"""Recovered historical prefetch invariant lives in the registered verifier."""

import subprocess

import pytest

from tessera.compiler.schedule_ir import SCHEDULE_MEMORY_SPACES
from tessera.compiler.scheduled_matmul import find_tessera_opt

pytestmark = pytest.mark.skipif(find_tessera_opt() is None, reason="requires native compiler")


@pytest.mark.parametrize("space", sorted(SCHEDULE_MEMORY_SPACES) + ["", "invented", "SHARED"])
def test_native_prefetch_memory_space_contract(space):
    source = f'''func.func @prefetch(%a: tensor<4xf32>) -> tensor<4xf32> {{
      %p = schedule.prefetch %a {{into = "{space}", overlap = "compute"}} : tensor<4xf32> -> tensor<4xf32>
      return %p : tensor<4xf32>
    }}'''
    result = subprocess.run([find_tessera_opt()], input=source, text=True, capture_output=True, timeout=30)
    if space in SCHEDULE_MEMORY_SPACES:
        assert result.returncode == 0, result.stderr
    else:
        assert result.returncode != 0
        assert "recognized destination memory space" in result.stderr
