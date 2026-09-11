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


def test_slotted_type_identity_includes_module():
    snapshots = []
    for module in ('left', 'right'):
        cls = type('Node', (), {'__slots__': ('value',), '__module__': module})
        node = cls()
        node.value = node
        snapshot = discover_objects(node, allow_slots=True)
        assert json.loads(snapshot.payloads[0]) == [
            'slotted_instance', module, 'Node', [module + '.Node:value']]
        assert snapshot.edges == ((0,),)
        snapshots.append(snapshot)
    assert snapshots[0].payloads != snapshots[1].payloads


@pytest.mark.parametrize('inherited', [False, True])
def test_slotted_module_must_be_plain_string(inherited):
    base = type('Base', (), {'__slots__': ('value',), '__module__': 123})
    cls = type('Node', (base,), {'__slots__': (), '__module__': 'valid'}) if inherited else base
    with pytest.raises(ValueError, match='module requires a plain string'):
        discover_objects(cls(), allow_slots=True)


def test_inherited_slot_owner_identity_includes_module():
    snapshots = []
    for module in ('left', 'right'):
        base = type('Base', (), {'__slots__': ('value',), '__module__': module})
        cls = type('Node', (base,), {'__slots__': (), '__module__': 'shared'})
        node = cls()
        node.value = node
        snapshots.append(discover_objects(node, allow_slots=True))
    assert snapshots[0].payloads != snapshots[1].payloads


def test_explicit_extension_layout_preserves_cycles_and_payload():
    from collections import deque
    from tessera.compiler.object_discovery import ExtensionLayout
    value = deque()
    value.append(value)
    with pytest.raises(ValueError):
        discover_objects(value)
    layout = ExtensionLayout(deque, 'deque-v1', lambda obj: (b'opaque', tuple(obj)))
    snapshot = discover_objects(value, extension_layouts=(layout,))
    assert snapshot.edges == ((0,),)
    assert json.loads(snapshot.payloads[0]) == ['extension', 'collections', 'deque', 'deque-v1', b'opaque'.hex()]
    with pytest.raises(ValueError, match='payload budget'):
        discover_objects(value, max_bytes=4, extension_layouts=(layout,))


@pytest.mark.parametrize('result', [(b'a', []), ('a', ()), [b'a', ()]])
def test_extension_extractor_result_is_checked(result):
    from collections import deque
    from tessera.compiler.object_discovery import ExtensionLayout
    with pytest.raises(ValueError, match='bytes and a tuple'):
        discover_objects(deque(), extension_layouts=(ExtensionLayout(deque, 'v1', lambda _: result),))


def test_extension_layout_does_not_capture_subclasses_or_override_builtins():
    from collections import deque
    from tessera.compiler.object_discovery import ExtensionLayout
    class Child(deque):
        __slots__ = ()
    def forbidden(_):
        raise AssertionError('wrong exact type')
    layout = ExtensionLayout(deque, 'v1', forbidden)
    with pytest.raises(ValueError):
        discover_objects(Child(), extension_layouts=(layout,))
    with pytest.raises(ValueError, match='builtin'):
        discover_objects([], extension_layouts=(ExtensionLayout(list, 'v1', forbidden),))


def test_general_mapping_keys_preserve_key_value_alias_and_cycles():
    key = (1, 2)
    mapping = {key: key}
    mapping['self'] = mapping
    snapshot = discover_objects(mapping)
    assert json.loads(snapshot.payloads[0]) == ['mapping']
    assert snapshot.edges[0][0] == snapshot.edges[0][1]
    assert snapshot.edges[0][-1] == 0


@pytest.mark.parametrize('factory', [set, frozenset])
def test_exact_set_discovery_does_not_rehash_elements(factory):
    class Key:
        def __hash__(self):
            if getattr(self, 'sealed', False):
                raise AssertionError('rehash')
            return 42
    key = Key()
    value = factory([key])
    key.sealed = True
    snapshot = discover_objects(value)
    assert len(snapshot.edges[0]) == 1


def test_builtin_subclass_hidden_payload_is_not_silently_dropped():
    from collections import deque
    class Queue(deque):
        pass
    value = Queue([1, 2])
    with pytest.raises(ValueError, match='builtin subclass payload'):
        discover_objects(value)


def test_custom_metaclass_cannot_run_comparison_hooks_during_discovery():
    class Meta(type):
        def __eq__(cls, other):
            raise AssertionError('metaclass comparison hook')
        def __hash__(cls):
            raise AssertionError('metaclass hash hook')
    class Value(metaclass=Meta):
        pass
    with pytest.raises(ValueError, match='custom metaclass'):
        discover_objects(Value())
