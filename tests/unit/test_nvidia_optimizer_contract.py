"""Host-free malformed-contract coverage for the generated CUDA optimizer ABI.

`kind` selects the update rule, so it is a semantic key and must fail closed
(Decision #21a). Before this coverage existed, `opt_k` routed every unrecognised
code into its trailing Adam/AdamW branch and the entry reported rc 1, so a
mis-mapped optimizer produced Adam updates that nothing downstream could
detect. The sibling entries in the same file already range-check their enums
(reduce `kind<0||kind>3`, MoE `kind<0||kind>2`).
"""
from __future__ import annotations

import numpy as np
import pytest

from tessera.compiler.emit import nvidia_cuda as nv


def _buffers(n=4):
    return [np.zeros(n, np.float32) for _ in range(4)]


@pytest.mark.parametrize("kind", [-1, 6, 99])
def test_optimizer_rejects_unknown_kind_before_cuda_compile(kind, monkeypatch):
    monkeypatch.setattr(
        nv, "_nvidia_cuda_compile_fn",
        lambda *_a, **_k: pytest.fail("must reject before CUDA compile"))
    p, g, m, v = _buffers()
    with pytest.raises(ValueError, match="optimizer kind"):
        nv.run_optimizer_f32(kind, p, g, m, v, 4, 0.1, 0.9, 0.99, 1e-8,
                             0.0, 1.0, 1.0)


def test_optimizer_kinds_match_the_runtime_name_map():
    from tessera.runtime import _OPTIMIZER_OPS
    assert {code for _, code, _ in _OPTIMIZER_OPS.values()} == nv._OPTIMIZER_KINDS


def test_optimizer_entry_range_checks_kind_like_its_siblings():
    source = nv._synthesize_optimizer_cuda()
    assert "if(kind<0||kind>5)return 2;" in source
