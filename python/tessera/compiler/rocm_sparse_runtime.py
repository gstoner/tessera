"""Opt-in logical sparse runtime binding with process-owned HIP resources.

This is an explicit compiler API, with an opt-in tracer/JIT adapter; not automatic sparse dispatch. No device
pointer crosses the boundary. Results escape only after validity verification and
successful teardown; uncertain workers remain retained until death is confirmed.
"""
from __future__ import annotations

import ctypes as ct
from dataclasses import dataclass, replace
import hashlib
import math
import multiprocessing as mp
import os
from pathlib import Path
import re
import subprocess
import threading

import numpy as np

from .native_driver_isolation import DriverIsolationLease, SpawnedProcessBoundary
from .native_gpu_storage import _decode_image
from .rocm_sparse_logical import sparse_logical_schedule_ir


@dataclass(frozen=True)
class SparseMatmulPackage:
    shape: tuple[int, int, int]
    dtype: str
    schedule_ir: str
    lowered_ir: str
    image: bytes
    compiler_digest: str
    accum: str = 'f32'
    digest: str = ''
    output_storage: str | None = None
    rhs_dtype: str | None = None
    integer_bits: int = 8
    native_graph_ir: str | None = None

    def _digest(self):
        parts = (self.native_graph_ir, repr(self.shape), self.dtype, self.rhs_dtype, self.integer_bits, self.accum, self.output_storage, self.schedule_ir, self.lowered_ir,
                 self.compiler_digest, hashlib.sha256(self.image).hexdigest())
        return hashlib.sha256(repr(parts).encode()).hexdigest()

    def validate(self):
        m,n,k = self.shape
        if max(m,n,k) > 256:
            raise ValueError('sparse runtime envelope is bounded to extents <= 256')
        if self.native_graph_ir is None and self.schedule_ir != sparse_logical_schedule_ir(m,n,k,self.dtype,accum=self.accum,output_storage=self.output_storage,rhs_dtype=self.rhs_dtype,integer_bits=self.integer_bits):
            raise ValueError('sparse runtime shape/storage disagrees with its Schedule program')
        if self.native_graph_ir is not None:
            mode = re.search(r'tessera.sparse_selection = "(checked_2to4|auto_2to4)"',self.schedule_ir)
            requested = re.search(r'tessera.sparse_policy = "(checked_2to4|auto_2to4)"',self.native_graph_ir)
            if mode is None or requested is None or mode[1] != requested[1]:
                raise ValueError('native sparse selection disagrees with logical policy')
            shape = re.search(r'tessera.sparse_shape = array<i64: (\d+), (\d+), (\d+)>',self.schedule_ir)
            storage = re.search(r'tessera.sparse_storage = "(f16|bf16)"',self.schedule_ir)
            output = re.search(r'tessera.sparse_output = "(f16|bf16|f32)"',self.schedule_ir)
            if (not self.native_graph_ir or shape is None or storage is None or output is None or
                tuple(map(int,shape.groups())) != self.shape or
                {'f16':'float16','bf16':'bfloat16'}[storage[1]] != self.dtype or
                output[1] != self.output_storage or self.accum != 'f32' or
                self.rhs_dtype not in (None,self.dtype) or self.integer_bits != 8):
                raise ValueError('native sparse descriptor disagrees with package ABI')
        if not self.image or self.digest != self._digest():
            raise ValueError('sparse runtime artifact identity disagrees')

    def run(self, a, b, *, device=0, timeout_seconds=30.0):
        """Return checked typed host output; never return a device allocation."""
        self.validate()
        if type(device) is not int or not 0 <= device < 2**31:
            raise ValueError('sparse runtime device requires a nonnegative ordinal')
        if type(timeout_seconds) not in (int,float) or not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError('sparse runtime timeout must be finite and positive')
        a,b = self._inputs(a,b)
        context = mp.get_context('spawn')
        parent,child = context.Pipe()
        process = context.Process(target=_worker,args=(child,self,a,b,device),daemon=True)
        try:
            process.start()
        except BaseException:
            parent.close(); child.close()
            raise
        child.close()
        lease = DriverIsolationLease(SpawnedProcessBoundary(process),
            context_identity=f'sparse-worker-{process.pid}',timeout_seconds=min(timeout_seconds,5.0))
        retained = False
        try:
            if not parent.poll(timeout_seconds):
                raise TimeoutError('sparse worker exceeded its execution deadline')
            message = parent.recv()
            process.join(timeout_seconds)
            if process.exitcode != 0:
                raise RuntimeError('sparse worker teardown is unconfirmed or failed')
            if message[0] == 'invalid':
                raise ValueError('sparse kernel refused invalid 2:4 input; no output exposed')
            if message[0] != 'result':
                raise RuntimeError(f'sparse worker failed: {message}')
            value = message[1]
            if not isinstance(value,np.ndarray) or value.shape != (self.shape[0],self.shape[1]) or value.dtype != self.result_dtype():
                raise RuntimeError('sparse worker result ABI disagrees')
            return value
        except BaseException:
            lease.mark_uncertain()
            owner = (process,parent,lease)
            try:
                lease.recover()
            except BaseException:
                # Keep the process/channel/lease for explicit late-exit recovery.
                with _RECOVERY_LOCK:
                    _UNCERTAIN.append(owner)
                retained = True
                raise RuntimeError('sparse worker death unconfirmed; ownership retained')
            raise
        finally:
            if not retained and process.exitcode is not None:
                parent.close()
                process.close()

    def result_dtype(self):
        import ml_dtypes
        return np.dtype({'f16':np.float16,'bf16':ml_dtypes.bfloat16}.get(
            self.output_storage or "",np.int32 if self.dtype in {'int8','uint8'} else np.float32))

    def _inputs(self,a,b):
        m,n,k = self.shape
        values = []
        for value,shape,dtype in ((a,(m,k),self.dtype),(b,(k,n),self.rhs_dtype or self.dtype)):
            value = np.asarray(value)
            if value.shape != shape or str(value.dtype) != dtype or not np.all(np.isfinite(value)):
                raise ValueError('sparse runtime requires finite matching logical matrix inputs')
            if self.integer_bits == 4:
                low, high = (-8,7) if dtype == 'int8' else (0,15)
                if np.any(value < low) or np.any(value > high):
                    raise ValueError('sparse INT4 input is outside its declared range')
            values.append(np.array(value,copy=True,order='C'))
        return tuple(values)


_UNCERTAIN: list = []
_RECOVERY_LOCK = threading.RLock()


def recover_sparse_workers():
    """Reconcile retained workers; a live worker is never treated as reclaimed."""
    with _RECOVERY_LOCK:
        for owner in list(_UNCERTAIN):
            process,connection,lease = owner
            lease.recover()
            if not lease.reusable:
                raise RuntimeError('sparse worker death remains unconfirmed')
            connection.close()
            process.close()
            _UNCERTAIN.remove(owner)


def compile_sparse_matmul(m, n, k, *, dtype='float16', accum='f32', output_storage=None, rhs_dtype=None, integer_bits=8, compiler=None, llvm_bin=None, toolkit=None):
    """Compile a bounded half/FP8/signed/unsigned-i8 logical matrix program for gfx1201."""
    source = sparse_logical_schedule_ir(m,n,k,dtype,accum=accum,output_storage=output_storage,rhs_dtype=rhs_dtype,integer_bits=integer_bits)
    return _compile_sparse_source(source,m,n,k,dtype=dtype,accum=accum,output_storage=output_storage,
        rhs_dtype=rhs_dtype,integer_bits=integer_bits,compiler=compiler,llvm_bin=llvm_bin,toolkit=toolkit)


def _compile_sparse_source(source,m,n,k,*,dtype,accum='f32',output_storage=None,rhs_dtype=None,
                           integer_bits=8,compiler=None,llvm_bin=None,toolkit=None,native_graph_ir=None):
    if max(m,n,k) > 256:
        raise ValueError('sparse runtime envelope is bounded to extents <= 256')
    from .scheduled_matmul import find_tessera_opt
    compiler = Path(compiler or find_tessera_opt() or '')
    llvm = Path(llvm_bin or os.environ['TESSERA_LLVM_BIN'])
    toolkit = str(toolkit or os.environ['ROCM_PATH'])
    env = {key: value for key,value in os.environ.items() if not key.startswith('ROCP') and key != 'LD_PRELOAD'}
    lowered = subprocess.check_output([str(compiler),'--tessera-schedule-to-tile',
        '--lower-tile-to-rocm','--lower-tessera-target-to-rocdl'],input=source,text=True,env=env,timeout=120)
    pipeline = 'builtin.module(gpu.module(convert-vector-to-llvm,convert-scf-to-cf,convert-gpu-to-rocdl,reconcile-unrealized-casts),rocdl-attach-target{chip=gfx1201},gpu-module-to-binary{toolkit='+toolkit+'})'
    binary = subprocess.check_output([str(llvm/'mlir-opt'),'--pass-pipeline='+pipeline],input=lowered,text=True,env=env,timeout=120)
    if binary.count('#gpu.object<') != 1:
        raise ValueError('sparse compilation requires one native image')
    image = _decode_image(re.findall(r'"((?:\\.|[^"\\])*)"',binary)[-1])
    package = SparseMatmulPackage((m,n,k),dtype,source,lowered,image,
                                 hashlib.sha256(compiler.read_bytes()).hexdigest(),accum=accum,output_storage=output_storage,rhs_dtype=rhs_dtype,integer_bits=integer_bits,native_graph_ir=native_graph_ir)
    package = replace(package,digest=package._digest())
    package.validate()
    return package


def _worker(connection, package, a, b, device):
    try:
        from tessera import runtime as rt
        hip = rt._load_hip_for_launch()
        if hip is None:
            raise RuntimeError('sparse runtime requires HIP')
        def check(status):
            if status:
                raise RuntimeError(f'sparse HIP operation failed with status {status}')
        check(hip.hipSetDevice(device))
        if rt._rocm_live_arch() != 'gfx1201':
            raise ValueError('sparse runtime requires the selected gfx1201 device')
        package.validate()
        m,n,k = package.shape
        P = ct.c_void_p
        module,fn,stream = P(),P(),P()
        blob = ct.create_string_buffer(package.image)
        check(hip.hipModuleLoadData(ct.byref(module),blob))
        check(hip.hipModuleGetFunction(ct.byref(fn),module,b'probe'))
        check(hip.hipStreamCreateWithFlags(ct.byref(stream),1))
        output = np.empty((m,n),package.result_dtype())
        status = np.zeros((m//16)*(n//16)*32,np.int32)
        hosts = (a,b,output,status)
        pointers = []
        for host in hosts:
            pointer = P()
            check(hip.hipMalloc(ct.byref(pointer),host.nbytes))
            pointers.append(pointer)
        for pointer,host in zip(pointers[:2],hosts[:2],strict=True):
            check(hip.hipMemcpyAsync(pointer,host.ctypes.data_as(P),host.nbytes,1,stream))
        # Missing or partially written validity words must refuse, not inherit
        # nonzero allocation contents from a previous GPU allocation.
        check(hip.hipMemsetAsync(pointers[3],0,status.nbytes,stream))
        values: list[ct.c_void_p | ct.c_int64] = []
        for pointer,host in zip(pointers,hosts,strict=True):
            values.extend((P(pointer.value),P(pointer.value),ct.c_int64(0),ct.c_int64(host.size),ct.c_int64(1)))
        args = (P*len(values))(*[ct.cast(ct.byref(value),P) for value in values])
        check(hip.hipModuleLaunchKernel(fn,(m//16)*(n//16),1,1,32,1,1,0,stream,args,None))
        check(hip.hipMemcpyAsync(status.ctypes.data_as(P),pointers[3],status.nbytes,2,stream))
        check(hip.hipStreamSynchronize(stream))
        valid = bool(np.all(status == 1))
        if valid:
            check(hip.hipMemcpyAsync(output.ctypes.data_as(P),pointers[2],output.nbytes,2,stream))
            check(hip.hipStreamSynchronize(stream))
        for pointer in reversed(pointers):
            check(hip.hipFree(pointer))
        check(hip.hipStreamDestroy(stream))
        check(hip.hipModuleUnload(module))
        connection.send(('result',output) if valid else ('invalid',))
        connection.close()
    except BaseException as error:
        try:
            connection.send(('error',type(error).__name__,str(error)))
            connection.close()
        finally:
            os._exit(1)  # No driver retry/destructor cleanup after uncertainty.
