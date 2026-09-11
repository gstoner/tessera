"""Compile serialized source exception tables to native allocation/root calls.

Static metadata and runtime tensor bytes. Each invocation owns a fresh heap; failed construction
closes it before publication. No Python exception constructor runs in native code.
"""
import ctypes as ct
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import threading
from typing import Any
from .native_exception_producer import NativeExceptionProducer
from .native_gpu_storage import _run
from .native_source_state import _attribute, bind_source_exception_types


def emit_exception_heap_ir(source, *, exception_types=None):
    contract = json.loads(_attribute(source, 'tessera.source_state'))
    bindings = bind_source_exception_types(contract, exception_types)
    from .source_exception_heap import validate_heap
    nodes, roots = validate_heap(contract['exception_heap'], bindings)
    if not nodes or any(n['unresolved'] for n in nodes):
        raise ValueError('native heap IR requires resolved exception payloads')
    dynamic = {i:n['args'][1] for i,n in enumerate(nodes) if len(n['args']) == 2 and n['args'][0] == '@tensor'}
    if dynamic and (not contract.get('error_dynamic') or any(site not in contract.get('error_payload_sites',()) for site in dynamic.values())):
        raise ValueError('native heap IR requires declared dynamic payload sites')
    payloads = [json.dumps(n, sort_keys=True).encode() for n in nodes]
    if sum(map(len,payloads)) > (1 << 20):
        raise ValueError('native exception IR payload exceeds one MiB')
    lines = ['module {']
    for i,payload in enumerate(payloads):
        if i in dynamic:
            continue
        text = ''.join('\\'+format(b,'02X') for b in payload)
        lines.append(f' llvm.mlir.global private constant @payload{i}("{text}") : !llvm.array<{len(payload)} x i8>')
    lines += [
        ' func.func private @tsr_exception_heap_alloc(!llvm.ptr, i32, !llvm.ptr, i32, i64, i64, i32, !llvm.ptr) -> i32',
        ' func.func private @tsr_exception_heap_edges(!llvm.ptr, i64, i64, i64) -> i32',
        ' func.func private @tsr_exception_heap_root(!llvm.ptr, i64, i32) -> i32',
        ' func.func @produce(%heap: !llvm.ptr, %handles: !llvm.ptr, %payloads: !llvm.ptr, %sizes: !llvm.ptr) -> i32 {',
        ' %zero = arith.constant 0 : i32', ' %one = arith.constant 1 : i32',
        ' %null = arith.constant 0 : i64']
    for i,payload in enumerate(payloads):
        lines += [f' %slot{i} = llvm.getelementptr %handles[{i}] : (!llvm.ptr) -> !llvm.ptr, i64',
                  f' %kind{i} = arith.constant {i} : i32']
        if i in dynamic:
            lines += [f' %pslot{i} = llvm.getelementptr %payloads[{i}] : (!llvm.ptr) -> !llvm.ptr, !llvm.ptr',
                      f' %p{i} = llvm.load %pslot{i} : !llvm.ptr -> !llvm.ptr',
                      f' %sslot{i} = llvm.getelementptr %sizes[{i}] : (!llvm.ptr) -> !llvm.ptr, i32',
                      f' %size{i} = llvm.load %sslot{i} : !llvm.ptr -> i32']
        else:
            lines += [f' %p{i} = llvm.mlir.addressof @payload{i} : !llvm.ptr',
                      f' %size{i} = arith.constant {len(payload)} : i32']
    count = 0
    def checked(call):
        nonlocal count
        lines.extend([f' %status{count} = {call}',
            f' %ok{count} = arith.cmpi eq, %status{count}, %zero : i32',
            f' cf.cond_br %ok{count}, ^next{count}, ^failed(%status{count} : i32)',
            f' ^next{count}:'])
        count += 1
    for i in range(len(nodes)):
        checked(f'func.call @tsr_exception_heap_alloc(%heap, %kind{i}, %p{i}, %size{i}, %null, %null, %one, %slot{i}) : (!llvm.ptr, i32, !llvm.ptr, i32, i64, i64, i32, !llvm.ptr) -> i32')
        lines.append(f' %handle{i} = llvm.load %slot{i} : !llvm.ptr -> i64')
    for i,node in enumerate(nodes):
        edges = [f'%handle{node[k]}' if node[k] is not None else '%null' for k in ('cause','context')]
        checked(f'func.call @tsr_exception_heap_edges(%heap, %handle{i}, {edges[0]}, {edges[1]}) : (!llvm.ptr, i64, i64, i64) -> i32')
    for i in range(len(nodes)):
        if i not in roots:
            checked(f'func.call @tsr_exception_heap_root(%heap, %handle{i}, %zero) : (!llvm.ptr, i64, i32) -> i32')
    lines += [' return %zero : i32', '^failed(%error: i32):', ' return %error : i32', ' }', '}']
    return '\n'.join(lines), tuple(roots), len(nodes), sum(len(p) for i,p in enumerate(payloads) if i not in dynamic)


class NativeExceptionIRProgram:
    def __init__(self, source, *, llvm_bin, runtime, exception_types=None):
        self._lock = threading.RLock()
        self._closed = False
        self.source = source
        self.contract = json.loads(_attribute(source, 'tessera.source_state'))
        self.nodes = self.contract['exception_heap']['nodes']
        self.dynamic = {i:n['args'][1] for i,n in enumerate(self.nodes)
                        if len(n['args']) == 2 and n['args'][0] == '@tensor'}
        self.mlir, self.roots, self.count, self.payload_bytes = emit_exception_heap_ir(source,exception_types=exception_types)
        llvm_bin, self.runtime = Path(llvm_bin), Path(runtime).resolve(strict=True)
        self.runtime_digest = hashlib.sha256(self.runtime.read_bytes()).hexdigest()
        lowered = _run(llvm_bin/'mlir-opt','--convert-scf-to-cf','--convert-arith-to-llvm',
                       '--convert-cf-to-llvm','--convert-func-to-llvm','--reconcile-unrealized-casts',source=self.mlir)
        self.llvm_ir = _run(llvm_bin/'mlir-translate','--mlir-to-llvmir',source=lowered)
        self._directory = tempfile.TemporaryDirectory(prefix='tessera-exception-ir-')
        path = Path(self._directory.name)
        (path/'producer.ll').write_text(self.llvm_ir)
        subprocess.run([str(llvm_bin/'clang'),'-shared','-fPIC','-O2',str(path/'producer.ll'),
                        str(self.runtime),'-Wl,-rpath,'+str(self.runtime.parent),'-o',str(path/'producer.so')],check=True,capture_output=True)
        self._library: Any = ct.CDLL(str(path/'producer.so'))
        self._call: Any = self._library.produce
        self._call.argtypes = [ct.c_void_p,ct.POINTER(ct.c_uint64),ct.POINTER(ct.c_void_p),ct.POINTER(ct.c_uint32)]
        self._call.restype = ct.c_int

    def instantiate(self, *, payloads=None, capacity=None, payload_capacity=None):
        with self._lock:
            if self._closed:
                raise ValueError("native exception IR program is closed")
            return self._instantiate(payloads=payloads,capacity=capacity,payload_capacity=payload_capacity)

    def _instantiate(self, *, payloads=None, capacity=None, payload_capacity=None):
        import numpy as np
        payloads = {} if payloads is None else payloads
        if not isinstance(payloads,dict) or set(payloads) != set(self.dynamic.values()):
            raise ValueError('native exception payload sites disagree with producer')
        arrays = {}
        for site,value in payloads.items():
            if not isinstance(value,np.ndarray) or value.dtype.kind not in 'biufc':
                raise ValueError('native exception payload requires numeric arrays')
            if value.nbytes > 1 << 20:
                raise ValueError('native exception payload exceeds one MiB')
            arrays[site] = np.array(value,copy=True,order='C')
        total = self.payload_bytes + sum(arrays[site].nbytes for site in self.dynamic.values())
        if total > 1 << 20:
            raise ValueError('native exception payload exceeds one MiB')
        pointers = (ct.c_void_p*self.count)()
        sizes = (ct.c_uint32*self.count)()
        for index,site in self.dynamic.items():
            pointers[index] = arrays[site].ctypes.data
            sizes[index] = arrays[site].nbytes
        if hashlib.sha256(self.runtime.read_bytes()).hexdigest() != self.runtime_digest:
            raise ValueError('native exception runtime identity changed')
        heap = NativeExceptionProducer(self.runtime,capacity=self.count if capacity is None else capacity,
            payload_capacity=max(1,total) if payload_capacity is None else payload_capacity)
        handles = (ct.c_uint64*self.count)()
        try:
            heap._check(self._call(heap._handle,handles,pointers,sizes))
            return heap, tuple(handles[i] for i in self.roots)
        except BaseException:
            heap.close()
            raise

    def decode(self, code, contract, outputs, bindings):
        from .source_exception_heap import decode_heap
        if any(contract.get(key) != self.contract.get(key) for key in ('exception_heap','error_dynamic','error_payload_sites')):
            raise ValueError('native exception decoder contract differs from producer')
        if type(code) is not int or not 1 <= code <= len(self.roots):
            raise ValueError('native exception root status is invalid')
        import numpy as np
        slots = contract.get('error_payload_sites',())
        payloads = {site:outputs[slots.index(site)-len(slots)].copy() for site in self.dynamic.values()}
        copied_outputs = list(outputs)
        heap,roots = self.instantiate(payloads=payloads)
        with heap:
            records = {}
            pending = [roots[code-1]]
            while pending:
                handle = pending.pop()
                if handle in records:
                    continue
                kind,payload,cause,context = heap.read(handle)
                if kind in self.dynamic:
                    node = dict(self.nodes[kind])
                    site = self.dynamic[kind]
                    value = payloads[site]
                    copied_outputs[slots.index(site)-len(slots)] = np.frombuffer(payload,dtype=value.dtype).reshape(value.shape).copy()
                else:
                    node = json.loads(payload)
                records[handle] = (node,cause,context)
                pending.extend(h for h in (cause,context) if h)
            indices = {handle:i for i,handle in enumerate(records)}
            nodes = []
            for node,cause,context in records.values():
                node['cause'] = indices[cause] if cause else None
                node['context'] = indices[context] if context else None
                nodes.append(node)
            copied = dict(schema=1,nodes=nodes,roots=[indices[roots[code-1]]])
        return decode_heap(copied,1,contract,copied_outputs,bindings)

    def close(self):
        import _ctypes
        with self._lock:
            if not self._closed:
                self._closed = True
                library = getattr(self,'_library',None)
                self._call = None
                self._library = None
                if library is not None:
                    _ctypes.dlclose(library._handle)
                if hasattr(self,'_directory'):
                    self._directory.cleanup()

    def __enter__(self):
        if self._closed:
            raise ValueError('native exception IR program is closed')
        return self

    def __exit__(self,*exc):
        self.close()

    def __del__(self):
        if hasattr(self,'_lock'):
            self.close()
