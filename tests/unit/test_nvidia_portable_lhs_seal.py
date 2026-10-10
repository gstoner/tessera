"""Typed JSON witness guards prevent Python equality from weakening a seal."""
import pytest
from tessera.compiler.prepared_nvidia_lhs import _same_json


@pytest.mark.parametrize("left,right",[
    ({"edge":{"m":1}},{"edge":{"m":True}}),
    ({"roles":[0,1]},{"roles":[False,True]}),
    ({"metric":0.0},{"metric":-0.0}),
    ({"policy":1},{"policy":1.0}),
    ({"args":["source","rhs"]},{"args":["rhs","source"]}),
])
def test_structurally_distinct_contracts_do_not_alias(left,right):
    assert not _same_json(left,right)


def test_reconstructed_plain_json_contract_reuses_its_witness():
    import json
    data={"roles":{"source":0,"rhs":1},"names":["source","rhs"],
          "metrics":{"spill":0,"median":1.25},"optional":None}
    assert _same_json(data,json.loads(json.dumps(data)))
