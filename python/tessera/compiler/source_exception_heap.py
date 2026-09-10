"""Indexed exception completion objects; no imports or source replay on decode.

The source producer is still bounded. Index edges remove recursive decoding and
allow shared/cyclic cause/context identities without duplicating constructors.
This is not a general Python object heap or a CPython frame representation.
"""
import builtins
import json
from typing import Any

BUILTINS = frozenset(('Exception','ValueError','RuntimeError','AssertionError',
                     'TypeError','IndexError','KeyError','OverflowError','ZeroDivisionError'))
LIMIT = 4096


def pack_exception_table(table):
    nodes: list[dict[str, Any]] = []
    roots: list[int] = []
    memo: dict[str, int] = {}
    def intern(edge):
        key=json.dumps(edge,sort_keys=True)
        if key in memo:return memo[key]
        if len(nodes)>=LIMIT:raise ValueError('source exception heap exceeds node capacity')
        index=len(nodes);memo[key]=index
        kind,args,*extra=edge
        node=dict(kind=kind,args=list(args),cause=None,context=None,suppress=False,location=None,unresolved=False)
        nodes.append(node)
        if extra and extra[0] is not None:
            cause,payload=extra[0];node['suppress']=True
            if cause=='binding':node['unresolved']=True
            elif cause=='edge':node['cause']=intern(payload)
            elif cause!='suppress':node['cause']=intern((cause,payload))
        if len(extra)>1:node['location']=extra[1]
        if len(extra)>2 and extra[2] is not None:node['context']=intern(extra[2])
        return index
    for edge in table:roots.append(intern(edge))
    return dict(schema=1,nodes=nodes,roots=roots)


def validate_heap(heap, bindings):
    if not isinstance(heap,dict) or type(heap.get('schema')) is not int or heap['schema']!=1:
        raise ValueError('invalid source exception heap schema')
    nodes,roots=heap.get('nodes'),heap.get('roots')
    if not isinstance(nodes,list) or len(nodes)>LIMIT or not isinstance(roots,list) or len(roots)>LIMIT:
        raise ValueError('invalid source exception heap capacity')
    def valid_index(value):return type(value) is int and 0<=value<len(nodes)
    if any(not valid_index(root) for root in roots):raise ValueError('invalid source exception heap root')
    for node in nodes:
        if not isinstance(node,dict) or set(node)!= {'kind','args','cause','context','suppress','location','unresolved'}:
            raise ValueError('invalid source exception heap node')
        kind,args=node['kind'],node['args']
        if type(kind) is not str or kind not in BUILTINS and kind not in bindings:
            raise ValueError('source exception class requires an explicit host binding')
        if not isinstance(args,list) or any(type(arg) is not str for arg in args):raise ValueError('invalid source exception heap payload')
        if any(node[key] is not None and not valid_index(node[key]) for key in ('cause','context')):
            raise ValueError('invalid source exception heap edge')
        if any(type(node[key]) is not bool for key in ('suppress','unresolved')):raise ValueError('invalid source exception heap flags')
        loc=node['location']
        if loc is not None and (not isinstance(loc,(list,tuple)) or len(loc)!=2 or type(loc[0]) is not str or type(loc[1]) is not int or loc[1]<1):
            raise ValueError('invalid source exception location')
    return nodes, roots


def decode_heap(heap, code, contract, outputs, bindings):
    # Snapshot before host constructors run: they may mutate caller-owned input.
    heap=json.loads(json.dumps(heap))
    nodes,roots=validate_heap(heap,bindings)
    if not 1<=code<=len(roots):raise RuntimeError('invalid native source exception status')
    root=roots[code-1];reachable=[];seen=set();pending=[root]
    slots=contract.get('error_payload_sites',())
    payloads={}
    while pending:
        index=pending.pop()
        if index in seen:continue
        seen.add(index);reachable.append(index);node=nodes[index]
        if node['unresolved']:raise RuntimeError('unresolved source exception reference')
        args=node['args']
        if len(args)==2 and args[0]=='@tensor':
            if not contract.get('error_dynamic') or slots and args[1] not in slots:
                raise RuntimeError('missing source exception payload slot')
            slot=slots.index(args[1])-len(slots) if slots else -1
            payloads[index]=(outputs[slot].copy(),)
        else:payloads[index]=tuple(args)
        pending.extend(node[key] for key in ('cause','context') if node[key] is not None)
    objects={}
    error: Any = None
    try:
        for index in reachable:
            kind=nodes[index]['kind']
            cls=bindings[kind] if kind in bindings else getattr(builtins,kind)
            error=cls(*payloads[index])
            if not isinstance(error,Exception):raise TypeError('source exception constructor returned a non-exception')
            objects[index]=error
    except BaseException:
        # A cached constructor traceback must not retain all earlier objects
        # from an unpublished heap through this decoder frame's locals.
        objects.clear()
        payloads.clear()
        error=None
        raise
    # Restore interpreter-owned exception fields without invoking arbitrary
    # custom __setattr__ or add_note hooks during completion publication.
    for index,error in objects.items():
        node=nodes[index]
        for field in ('cause', 'context'):
            value = objects[node[field]] if node[field] is not None else None
            BaseException.__dict__['__' + field + '__'].__set__(error, value)
        BaseException.__dict__['__suppress_context__'].__set__(error, node['suppress'])
        if node['location'] is not None:
            file,line=node['location']
            attrs = BaseException.__dict__['__dict__'].__get__(error)
            notes = attrs.get('__notes__')
            if notes is None:
                notes = []
                attrs['__notes__'] = notes
            if not isinstance(notes, list):
                raise TypeError('source exception notes must be a list')
            list.append(notes, f'Native source raise at {file}:{line}; no Python frame executed there')
            attrs['__tessera_native_frames__'] = ({
                'schema': 1, 'file': file, 'line': line,
                'function': contract.get('function_name', '<native>'),
                'instruction': contract.get('instruction_sites', {}).get(str(line)),
            },)
    return objects[root]
