"""F5 collective adjoints are wired into the paired backward ABI."""

from __future__ import annotations

import subprocess

import pytest

from tessera import _jit_boundary as jb


def _run(op: str) -> str:
    opt = jb._find_tessera_opt()
    if opt is None:
        pytest.skip("tessera-opt not built")
    text = f'''module {{
  func.func @collective(%x: tensor<4x8xf32>) -> tensor<4x8xf32>
      attributes {{tessera.autodiff = "reverse"}} {{
    %y = "tessera.{op}"(%x) {{axis = "dp", op = "sum"}} :
      (tensor<4x8xf32>) -> tensor<4x8xf32>
    return %y : tensor<4x8xf32>
  }}
}}
'''
    result = subprocess.run(
        [str(opt), "--tessera-autodiff-paired", "/dev/stdin"],
        input=text, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


@pytest.mark.parametrize(
    "forward,backward",
    [
        ("all_reduce", "tessera.all_reduce"),
        ("all_gather", "tessera.reduce_scatter"),
        ("reduce_scatter", "tessera.all_gather"),
    ],
)
def test_collective_transpose_is_emitted_in_paired_backward(forward, backward):
    text = _run(forward)
    bwd = text.split("func.func @collective__bwd", 1)[1]
    assert backward in bwd
    assert 'axis = "dp"' in bwd

