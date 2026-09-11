import json
import pytest
from tessera.compiler.object_discovery import discover_objects


def test_cycles_aliases_and_plain_instances_without_hooks():
    class Node:
        def __getattribute__(self, name):
            raise AssertionError("user hook called")

    node = Node()
    fields = object.__getattribute__(node, "__dict__")
    value = [node]
    fields.update(left=value, right=value)
    snapshot = discover_objects(node)
    assert snapshot.roots == (0,)
    assert snapshot.edges == ((1, 1), (0,))
    assert json.loads(snapshot.payloads[0])[0] == "instance"


def test_custom_descriptor_is_not_executed():
    class Dangerous:
        @property
        def __dict__(self):
            raise AssertionError("descriptor executed")

    with pytest.raises(ValueError, match="descriptors"):
        discover_objects(Dangerous())


@pytest.mark.parametrize(
    "value,options,reason",
    [
        ([1, 2], {"max_nodes": 2}, "node"),
        ([1, 2], {"max_edges": 1}, "reference"),
        ("long", {"max_bytes": 2}, "payload"),
        ({1: 2}, {}, "string keys"),
        (float("nan"), {}, "finite"),
    ],
)
def test_discovery_fails_closed(value, options, reason):
    with pytest.raises(ValueError, match=reason):
        discover_objects(value, **options)
