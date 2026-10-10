"""Bias roles are validated before native allocation or module import."""
import json
from pathlib import Path
import numpy as np
import pytest
from tessera.compiler import native_attention_jvp_runtime as module

ROOT=Path(__file__).resolve().parents[2]
ARTIFACTS=ROOT/"benchmarks/baselines/nvidia_bias_jvp_20261006/public-artifacts"

@pytest.fixture(params=[
    "k5_c0_1x4x1x5_bias_v_k_q","k129_c1_2x4x3x129_bias",
    "k129_c1_1x4x1x129_v",
])
def owner(request,monkeypatch):
    raw=json.loads((ARTIFACTS/(request.param+".json")).read_text())
    metadata=raw["native_jvp"]["steps"][0]["child_metadata"]
    service=module.PreparedAttentionJVP(metadata)
    def forbidden(*args,**kwargs):
        raise AssertionError("native allocation before host contract validation")
    monkeypatch.setattr(service,"_prepare",forbidden)
    return service,metadata

def test_every_biased_primal_and_active_direction_is_checked(owner):
    service,metadata=owner
    assert service.biased and service.count==4
    for index in range(len(service.shapes)):
        values=[np.zeros(shape,np.float32) for shape in service.shapes]
        values[index]=values[index].astype(np.float16)
        with pytest.raises(ValueError,match="storage"):
            service.invoke(metadata,values)
        values=[np.zeros(shape,np.float32) for shape in service.shapes]
        values[index]=values[index].reshape(-1)
        with pytest.raises(ValueError,match="storage"):
            service.invoke(metadata,values)

def test_biased_missing_direction_and_changed_names_are_checked(owner):
    service,metadata=owner
    values=[np.zeros(shape,np.float32) for shape in service.shapes]
    with pytest.raises(ValueError,match="arity"):
        service.invoke(metadata,values[:-1])
    changed=dict(metadata,arg_names=metadata["arg_names"][:-1]+["wrong_bias_role"])
    with pytest.raises(ValueError,match="launch names"):
        service.invoke(changed,values)
