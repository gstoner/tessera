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


def test_declared_slots_preserve_inherited_cycles_without_hooks():
    class Base:
        __slots__ = ('left',)
    class Node(Base):
        __slots__ = ('right', 'unset')
        def __getattribute__(self, name):
            raise AssertionError('user hook called')
    node = Node()
    object.__setattr__(node, 'left', node)
    object.__setattr__(node, 'right', node)
    with pytest.raises(ValueError, match='declared native layout'):
        discover_objects(node)
    snapshot = discover_objects(node, allow_slots=True)
    assert snapshot.edges == ((0, 0),)
    assert json.loads(snapshot.payloads[0])[0] == 'slotted_instance'


def test_shadowed_slot_property_is_never_called():
    class Node:
        __slots__ = ('value',)
    Node.value = property(lambda _: (_ for _ in ()).throw(AssertionError('hook called')))
    with pytest.raises(ValueError, match='native member'):
        discover_objects(Node(), allow_slots=True)


def test_slots_cannot_silently_drop_an_instance_dictionary():
    class Base:
        pass
    class Node(Base):
        __slots__ = ('value',)
    node = Node()
    node.extra = [node]
    with pytest.raises(ValueError, match='fully declared'):
        discover_objects(node, allow_slots=True)
