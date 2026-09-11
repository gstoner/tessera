"""Bounded GPU exception-payload frame allocation from native source metadata.

One synchronous writer allocates all numeric site payloads transactionally from
caller-owned GPU storage. Generation/offset/length records are published only on
success. This is frame reclamation, not an arbitrary object graph collector.
"""
from dataclasses import dataclass
import inspect
import hashlib
from typing import Any
import json
from pathlib import Path
from .native_gpu_storage import build_native_gpu_storage, _run, NativeGPUStoragePackage
from .native_gpu_tensor import TensorSpec, IndexSpec
from .native_storage_contract import attach_tensor_contract, generate_tensor_binding
from .native_source_state import _attribute, bind_source_exception_types
from .source_exception_heap import validate_heap


def emit_gpu_exception_heap(source, capacity):
    if type(capacity) is not int or not 1 <= capacity <= 262144:
        raise ValueError('GPU exception payload capacity must be bounded positive f32 elements')
    contract = json.loads(_attribute(source,'tessera.source_state'))
    nodes,_ = validate_heap(contract['exception_heap'],bind_source_exception_types(contract))
    if not nodes or any(n['unresolved'] or len(n['args']) != 2 or n['args'][0] != '@tensor' for n in nodes):
        raise ValueError('GPU heap requires resolved numeric payload nodes')
    sites = contract.get('error_payload_sites',[])
    if not contract.get('error_dynamic') or any(n['args'][1] not in sites for n in nodes):
        raise ValueError('GPU heap payload sites are undeclared')
    # A frame contains one value per site; cause/context topology stays in the
    # serialized source contract. No Python object address crosses the ABI.
    count = len(nodes)
    if len(set(n['args'][1] for n in nodes)) != count or len(sites) != count:
        raise ValueError('GPU heap requires one node per declared payload site')
    specs = (TensorSpec('lengths','int64',(count,)),TensorSpec('input','fp32',(capacity,)),
             TensorSpec('payload','fp32',(capacity,),True),TensorSpec('records','int64',(count,3),True),
             TensorSpec('status','int64',(2,),True),IndexSpec('generation',1,(1<<31)-1),IndexSpec('scratch',1,1))
    text = f'''module {{
gpu.module @native_tape {{
gpu.func @product(%lengths: !llvm.ptr<1>, %input: !llvm.ptr<1>, %payload: !llvm.ptr<1>, %records: !llvm.ptr<1>, %status: !llvm.ptr<1>, %generation: index, %scratch: index) kernel attributes {{known_block_size = array<i32: 1, 1, 1>}} {{
%marker = memref.alloca(%scratch) : memref<?xf32>
"tile.alloc_shared"(%marker) : (memref<?xf32>) -> ()
%zero = arith.constant 0 : i64
%one = arith.constant 1 : i64
%three = arith.constant 3 : i64
%count = arith.constant {count} : i64
%capacity = arith.constant {capacity} : i64
%iz = arith.constant 0 : index
%io = arith.constant 1 : index
%icount = arith.constant {count} : index
%yes = arith.constant true
%sum:2 = scf.for %iv = %iz to %icount step %io iter_args(%total = %zero, %valid = %yes) -> (i64, i1) {{
%i = arith.index_cast %iv : index to i64
%lp = llvm.getelementptr %lengths[%i] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, i64
%len = llvm.load %lp : !llvm.ptr<1> -> i64
%nonnegative = arith.cmpi sge, %len, %zero : i64
%room = arith.subi %capacity, %total : i64
%fits = arith.cmpi ule, %len, %room : i64
%ok = arith.andi %nonnegative, %fits : i1
%nextvalid = arith.andi %valid, %ok : i1
%added = arith.addi %total, %len : i64
%bounded = arith.select %ok, %added, %total : i64
scf.yield %bounded, %nextvalid : i64, i1
}}
%usedp = llvm.getelementptr %status[1] : (!llvm.ptr<1>) -> !llvm.ptr<1>, i64
scf.if %sum#1 {{
%epoch = arith.index_cast %generation : index to i64
%done = scf.for %iv = %iz to %icount step %io iter_args(%offset = %zero) -> i64 {{
%i = arith.index_cast %iv : index to i64
%lp = llvm.getelementptr %lengths[%i] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, i64
%len = llvm.load %lp : !llvm.ptr<1> -> i64
%end = arith.addi %offset, %len : i64
scf.for %j = %offset to %end step %one : i64 {{
%src = llvm.getelementptr %input[%j] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
%dst = llvm.getelementptr %payload[%j] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
%value = llvm.load %src : !llvm.ptr<1> -> f32
llvm.store %value, %dst : f32, !llvm.ptr<1>
}}
%base = arith.muli %i, %three : i64
%rp = llvm.getelementptr %records[%base] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, i64
%op = llvm.getelementptr %rp[1] : (!llvm.ptr<1>) -> !llvm.ptr<1>, i64
%np = llvm.getelementptr %rp[2] : (!llvm.ptr<1>) -> !llvm.ptr<1>, i64
llvm.store %epoch, %rp : i64, !llvm.ptr<1>
llvm.store %offset, %op : i64, !llvm.ptr<1>
llvm.store %len, %np : i64, !llvm.ptr<1>
scf.yield %end : i64
}}
llvm.store %done, %usedp : i64, !llvm.ptr<1>
llvm.store %zero, %status : i64, !llvm.ptr<1>
}} else {{
llvm.store %zero, %usedp : i64, !llvm.ptr<1>
llvm.store %one, %status : i64, !llvm.ptr<1>
}}
gpu.return
}} }} }}'''
    encoded = ''.join('\\'+format(b,'02X') for b in json.dumps(contract,sort_keys=True).encode())
    ir = attach_tensor_contract(text,specs,grid=(1,1,1),block=(1,1,1))
    ir = ir.replace('module attributes {','module attributes {tessera.exception_heap.source = "'+encoded+'", ',1)
    return ir,specs


@dataclass(frozen=True)
class GPUExceptionHeap:
    source: str
    capacity: int
    compiler: Path
    package: NativeGPUStoragePackage

    def validate(self):
        self.package.validate()
        if hashlib.sha256(self.compiler.read_bytes()).hexdigest() != self.package.compiler_digest:
            raise ValueError('GPU heap compiler identity changed')
        ir,specs = emit_gpu_exception_heap(self.source,self.capacity)
        replay = _run(self.compiler,'--allow-unregistered-dialect','--tessera-tile-buffer-reuse',
                      '--tessera-tile-buffer-arena','--canonicalize',source=ir)
        if replay != self.package.arena_ir:
            raise ValueError('GPU exception heap disagrees with source replay')
        return specs

    def decode_completed(self, code, status, records, payload, *, generation):
        """Decode copied-back buffers after the caller proves GPU completion."""
        import numpy as np
        from .source_exception_heap import decode_heap
        contract = json.loads(_attribute(self.source,'tessera.source_state'))
        nodes,roots = validate_heap(contract['exception_heap'],{})
        if type(generation) is not int or not 1 <= generation < (1<<31) or type(code) is not int or not 1 <= code <= len(roots):
            raise ValueError('GPU heap generation/root is invalid')
        if (not isinstance(status,np.ndarray) or status.dtype != np.int64 or status.shape != (2,)
                or not isinstance(records,np.ndarray) or records.dtype != np.int64 or records.shape != (len(nodes),3)
                or not isinstance(payload,np.ndarray) or payload.dtype != np.float32 or payload.shape != (self.capacity,)):
            raise ValueError('GPU heap completion ABI disagrees')
        status,records,payload = status.copy(),records.copy(),payload.copy()
        if status[0] != 0:
            raise MemoryError('GPU exception payload frame allocation failed')
        outputs: list[Any] = [None]*len(nodes)
        end = 0
        for node,record in zip(nodes,records,strict=True):
            epoch,offset,length = map(int,record)
            if epoch != generation or offset != end or length < 0 or offset+length > self.capacity:
                raise ValueError('GPU heap stale generation or invalid payload range')
            end += length
            site = contract['error_payload_sites'].index(node['args'][1])
            outputs[site] = payload[offset:end].copy()
        if status[1] != end:
            raise ValueError('GPU heap published size disagrees')
        return decode_heap(contract['exception_heap'],code,contract,outputs,{})

    def bind(self):
        specs = self.validate()
        signature = inspect.Signature([inspect.Parameter(s.name,inspect.Parameter.POSITIONAL_ONLY) for s in specs])
        return generate_tensor_binding(self.package,signature)


def materialize_gpu_exception_heap(source, *, capacity, compiler, llvm_bin, backend, chip):
    ir,_ = emit_gpu_exception_heap(source,capacity)
    package = build_native_gpu_storage(ir,compiler=Path(compiler),llvm_bin=Path(llvm_bin),backend=backend,chip=chip)
    result = GPUExceptionHeap(source,capacity,Path(compiler),package)
    result.validate()
    return result
