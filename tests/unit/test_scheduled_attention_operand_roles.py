"""Native Schedule SSA, rather than function position, owns attention roles."""
import pytest
from tessera.compiler.scheduled_attention import schedule_attention_argument_types


def source(backward=False):
    name="flash_attn_bwd" if backward else "flash_attn"
    schedule="attention_backward" if backward else "attention"
    return f"""module {{
  func.func @permuted(%arg0: tensor<1x1x5x3xf32>, %arg1: tensor<1x2x3x4xf32>, %arg2: tensor<1x1x5x4xf32>) -> tensor<1x2x3x3xf32> {{
    %0 = tessera.{name} %arg1, %arg2, %arg0 {{}} : () -> tensor<1x2x3x3xf32>
    %1 = schedule.{schedule} %0 {{}} : tensor<1x2x3x3xf32> -> tensor<1x2x3x3xf32>
    return %1 : tensor<1x2x3x3xf32>
  }}
}}"""


@pytest.mark.parametrize("backward",[False,True])
def test_roles_follow_retained_graph(backward):
    assert schedule_attention_argument_types(source(backward),backward=backward)==(
        "tensor<1x2x3x4xf32>","tensor<1x1x5x4xf32>","tensor<1x1x5x3xf32>")


@pytest.mark.parametrize("old,new",[
    ("schedule.attention %0","schedule.attention %unrelated"),
    ("%arg1, %arg2, %arg0","%arg1, %arg1, %arg0"),
    ("%arg1, %arg2, %arg0","%arg1, %arg2, %unknown"),
])
def test_malformed_role_lineage_rejected(old,new):
    with pytest.raises(ValueError):
        schedule_attention_argument_types(source().replace(old,new))
