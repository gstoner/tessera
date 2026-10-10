"""Native saved-LSE JVP Graph/Schedule contracts preserve both result roles."""
from dataclasses import replace
import json
import re

import numpy as np
import pytest
import tessera as ts

from tessera.compiler.native_gpu_storage import _decode_image
from tessera.compiler.scheduled_matmul import run_tessera_opt
from tests.device.nvidia.test_resident_attention_forward import biased, saved


def _source(with_lse, wrt):
    fn = ts.jit(target="nvidia_sm120", autodiff="forward", wrt=wrt)(
        saved if with_lse else biased)
    inputs = (np.ones((1, 1, 5, 3), np.float32),
              np.ones((1, 2, 3, 4), np.float32),
              np.ones((1, 1, 5, 4), np.float32),
              np.ones((1, 2, 1, 5), np.float32))
    module = fn._specialized_autodiff_module(inputs, {})
    module = replace(module, module_attrs={**module.module_attrs,
                     "tessera.target": '"nvidia_sm120"', "tessera.arch": '"sm_120"'})
    return re.sub(r'=\s+(tessera\.[A-Za-z0-9_.]+)\(', r'= "\1"(', module.to_mlir())


@pytest.mark.parametrize("with_lse", [False, True])
@pytest.mark.parametrize("wrt", [("v",), ("q", "k"), ("bias",)])
def test_native_saved_lse_jvp_result_selection_is_hashed(production_compiler, with_lse, wrt):
    product = run_tessera_opt(production_compiler, _source(with_lse, wrt),
                             "--tessera-autodiff-forward=export-attention-jvp")
    match = re.search(r'tessera.autodiff.attention_jvp_contract = "((?:\\.|[^"\\])*)"', product)
    assert match is not None
    request = json.loads(_decode_image(match[1]))
    assert request["schema"] == (3 if with_lse else 2)
    assert request.get("saved_lse", False) is with_lse
    lowered = run_tessera_opt(production_compiler, product,
                             "--pass-pipeline=builtin.module(tessera-graph-to-schedule,tessera-schedule-to-tile)")
    match = re.search(r'tessera.native_tensor_contract = "((?:\\.|[^"\\])*)"', lowered)
    assert match is not None
    manifest = json.loads(_decode_image(match[1]))
    results = [row for row in manifest["arguments"] if row.get("writable")]
    assert [row["name"] for row in results] == (["tangent", "dlse"] if with_lse else ["tangent"])
    assert results[0]["shape"] == [1, 2, 3, 3]
    if with_lse:
        assert results[1]["shape"] == [1, 2, 3]
        assert "saved_lse = true" in lowered
        assert "saved_lse_jvp_with_lse" in lowered
    else:
        assert "saved_lse_jvp_with_lse" not in lowered
    assert manifest["block"] == [128, 1, 1]
