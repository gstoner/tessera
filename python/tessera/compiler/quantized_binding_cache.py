"""Bounded, complete-content certificates for immutable compiler packages.

Only host validation and ABI construction are memoized. Runtime inputs,
readiness, device identity and native execution are checked on every call.
"""
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import fields,is_dataclass
from functools import wraps
import os
import struct
import threading

_ENTRIES=16
_ENTRY_BYTES=4*1024*1024


class _Uncacheable(Exception):
    pass


def _snapshot(value):
    remaining=_ENTRY_BYTES
    active=set()
    def freeze(item):
        nonlocal remaining
        remaining-=32
        if remaining<0:raise _Uncacheable
        kind=type(item)
        if item is None or kind is bool:return (kind,item)
        if kind is float:return (kind,struct.pack("!d",item))
        if kind is int:
            remaining-=(item.bit_length()+7)//8
            if remaining<0:raise _Uncacheable
            return (kind,item)
        if kind in (str,bytes):
            remaining-=len(item) if kind is bytes or item.isascii() else len(item.encode("utf-8"))
            if remaining<0:raise _Uncacheable
            return (kind,item)
        identity=id(item)
        if identity in active:raise _Uncacheable
        active.add(identity)
        try:
            if is_dataclass(item) and not isinstance(item,type):
                params=getattr(item,"__dataclass_params__",None)
                if params is None or not params.frozen:raise _Uncacheable
                return (kind,tuple((freeze(field.name),freeze(getattr(item,field.name))) for field in fields(item)))
            if kind in (tuple,list):return (kind,tuple(freeze(v) for v in item))
            if isinstance(item,Mapping) and all(type(k) is str for k in item):
                return (kind,tuple((freeze(k),freeze(item[k])) for k in sorted(item)))
            raise _Uncacheable
        finally:
            active.remove(identity)
    return freeze(value)


def _memoize(package_position):
    def decorate(method):
        cache: OrderedDict[object,object]=OrderedDict()
        lock=threading.RLock();process=os.getpid()
        hits=misses=0
        def ensure_process():
            nonlocal process,cache,lock,hits,misses
            if os.getpid()!=process:
                cache=OrderedDict();lock=threading.RLock();process=os.getpid()
                hits=misses=0
        @wraps(method)
        def wrapped(*args,**kwargs):
            nonlocal hits,misses
            # Reset even inspection locks before touching inherited state.
            ensure_process()
            if os.environ.get('TESSERA_ROCM_PROGRAM_BINDING_CACHE','1')!='1':
                return method(*args,**kwargs)
            package=args[package_position] if len(args)>package_position else kwargs.get('self' if package_position==0 else 'package')
            try:key=_snapshot(package)
            except _Uncacheable:return method(*args,**kwargs)
            with lock:
                if key in cache:
                    hits+=1;cache.move_to_end(key)
                    return cache[key]
                misses+=1
                result=method(*args,**kwargs)
                try:unchanged=key==_snapshot(package)
                except _Uncacheable:unchanged=False
                if not unchanged:
                    raise ValueError('native quantized package changed during validation or ABI binding')
                cache[key]=result
                if len(cache)>_ENTRIES:cache.popitem(last=False)
                return result
        def clear():
            nonlocal hits,misses
            ensure_process()
            with lock:cache.clear();hits=misses=0
        def info():
            ensure_process()
            with lock:return {'hits':hits,'misses':misses,'size':len(cache),'maxsize':_ENTRIES,'entry_byte_limit':_ENTRY_BYTES}
        setattr(wrapped,"cache_clear",clear)
        setattr(wrapped,"cache_info",info)
        return wrapped
    return decorate


memoized_validation=_memoize(0)
memoized_binding=_memoize(1)
