"""Indexed completion graphs preserve identity without recursive materialization."""
import numpy as np
import pytest
from tessera.compiler.native_source_state import decode_source_exception
from tessera.compiler.source_exception_heap import pack_exception_table


def node(**changes):
    return dict(kind='ValueError',args=['value'],cause=None,context=None,
                suppress=False,location=None,unresolved=False,**changes) if not changes else {**node(),**changes}


def decode(nodes,bindings=None):
    return decode_source_exception({'exception_heap':dict(schema=1,nodes=nodes,roots=[0])},
                                   [np.array([1],np.float32)],exception_types=bindings)


def test_heap_preserves_shared_and_cyclic_identity():
    error=decode([node(cause=1,context=1,suppress=True),node(context=0)])
    assert error.__cause__ is error.__context__
    assert error.__cause__.__context__ is error
    assert error.__suppress_context__ is True


def test_transport_preserves_typed_native_frame_record_without_faking_traceback():
    contract={'exception_heap':dict(schema=1,nodes=[node(location=('model.py',17))],roots=[0]),
              'function_name':'step','instruction_sites':{'17':23}}
    error=decode_source_exception(contract,[np.array([1],np.float32)])
    assert error.__traceback__ is None
    assert error.__tessera_native_frames__ == ({'schema':1,'file':'model.py','line':17,
                                                'function':'step','instruction':23},)


def test_long_heap_chain_does_not_use_python_recursion():
    count=1500
    error=decode([node(cause=i+1 if i+1<count else None) for i in range(count)])
    seen=0
    while error is not None:
        seen+=1;error=error.__cause__
    assert seen==count


@pytest.mark.parametrize('bad', [True,-1,2,1.0,'1'])
def test_invalid_heap_edge_precedes_constructors(bad):
    calls=[]
    class Custom(Exception):
        def __init__(self,*args):calls.append(args)
    with pytest.raises(ValueError,match='edge'):
        decode([node(kind='Custom',cause=bad),node()],{'Custom':Custom})
    assert calls==[]


def test_producer_interns_shared_occurrences_not_equal_literal_instances():
    child=('ValueError',['same'],None,['f.py',1],None,'site:0')
    other=('ValueError',['same'],None,['f.py',1],None,'site:1')
    heap=pack_exception_table([('RuntimeError',['root'],('edge',child),['f.py',2],child,'root'),other])
    root=heap['nodes'][heap['roots'][0]]
    assert root['cause']==root['context']
    assert heap['roots'][1]!=root['cause']


def test_constructor_mutation_cannot_change_validated_heap():
    nodes=[node(kind='Custom',cause=1),node()]
    class Custom(Exception):
        def __init__(self,*args):nodes[1]['kind']='Missing'
    error=decode(nodes,{'Custom':Custom})
    assert type(error.__cause__) is ValueError


def test_failed_materialization_does_not_cache_unpublished_objects():
    import weakref
    references=[]
    class First(Exception):
        pass
    class Broken(Exception):
        def __init__(self,*args):raise RuntimeError('constructor failed')
    class Recorded(First):
        def __init__(self,*args):
            super().__init__(*args)
            references.append(weakref.ref(self))
    nodes=[node(kind='Recorded',cause=1),node(kind='Broken')]
    with pytest.raises(RuntimeError,match='constructor failed') as caught:
        decode(nodes,{'Recorded':Recorded,'Broken':Broken})
    # Keep the failure traceback alive, as the public completion cache does.
    assert caught.value.__traceback__ is not None
    assert references[0]() is None


def test_native_completion_bypasses_custom_exception_field_hooks():
    calls = []
    class Custom(Exception):
        def __setattr__(self, name, value):
            calls.append(name)
            raise RuntimeError('unexpected user field hook')
        def add_note(self, note):
            raise RuntimeError('unexpected user note hook')
    error = decode([node(kind='Custom',cause=1,context=1,suppress=True,
                         location=('step.py',9)),node(context=0)], {'Custom':Custom})
    assert calls == []
    assert error.__cause__ is error.__context__
    assert error.__cause__.__context__ is error
    assert error.__suppress_context__
    assert error.__tessera_native_frames__[0]['line'] == 9
    assert error.__traceback__ is None


def test_native_chain_clears_constructor_edges_and_bypasses_descriptors():
    class Custom(Exception):
        def __init__(self, *args):
            super().__init__(*args)
            BaseException.__dict__['__cause__'].__set__(self,ValueError('unrecorded'))
        @property
        def __context__(self):
            raise AssertionError('custom context descriptor read')
        @__context__.setter
        def __context__(self, value):
            raise AssertionError('custom context descriptor written')
    error = decode([node(kind='Custom')],{'Custom':Custom})
    assert BaseException.__dict__['__cause__'].__get__(error) is None
    assert BaseException.__dict__['__context__'].__get__(error) is None
