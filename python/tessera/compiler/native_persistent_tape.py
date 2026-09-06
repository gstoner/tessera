"""Split compiler-produced AD products with persistent CUDA/HIP residual storage.

The initial physical envelope is static f32 tensors and bounded for/if regions,
executed serially on one GPU thread. Nested inner states may still be replayed
by the compiler's backward program; exported residuals persist across calls.
"""
from dataclasses import dataclass
import ctypes as ct
import hashlib
import inspect
import json
from pathlib import Path
import re
import threading
from .native_gpu_storage import _run, _decode_image, build_native_gpu_storage, NativeGPUStoragePackage
from .native_gpu_tensor import TensorSpec, IndexSpec
from .native_storage_contract import attach_tensor_contract, generate_tensor_binding, read_tensor_contract, tensor_contract_specs
from .native_device_tape import _Buffer


def _attribute(text,name):
    fields=re.findall(re.escape(name)+r' = "((?:\\.|[^"\\])*)"',text)
    if len(fields)!=1:
        raise ValueError('split tape requires unique compiler ABI and lineage')
    return _decode_image(fields[0]).decode()


def _shape(type_name):
    match=re.fullmatch(r'tensor<((?:[1-9][0-9]*x)*)f32>',type_name)
    if not match:
        raise ValueError('persistent GPU tape currently requires static f32 tensor slots')
    shape=tuple(int(x) for x in match[1].split('x') if x)
    count=1
    for dim in shape:
        if dim>1024 or count>1024//dim:
            raise ValueError('persistent GPU tape slot exceeds 1024 elements')
        count*=dim
    return shape


def materialize_persistent_tape(source, *, compiler, llvm_bin, backend, chip):
    """Generate, bufferize and materialize both products from one fresh request."""
    compiler,llvm_bin=Path(compiler),Path(llvm_bin)
    if backend not in ('nvidia','rocm'):
        raise ValueError('persistent tape requires a CUDA or HIP consumer')
    packages=[]
    contracts=[]
    lineages=[]
    for role in ('forward','backward'):
        exported=_run(compiler,'--tessera-autodiff-paired=export-product='+role,source=source)
        contract=json.loads(_attribute(exported,'tessera.autodiff.product_abi'))
        contracts.append(contract)
        lineages.append(_attribute(exported,'tessera.autodiff.product_pair'))
        shapes=[_shape(t) for t in contract['inputs']+contract['results']]
        native=_run(compiler,'--tessera-to-linalg',source=exported)
        buffered=_run(llvm_bin/'mlir-opt','--allow-unregistered-dialect',
            '--convert-elementwise-to-linalg',
            '--one-shot-bufferize=bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map allow-return-allocs-from-loops',
            '--buffer-results-to-out-params=modify-public-functions hoist-static-allocs',
            '--convert-linalg-to-loops','--canonicalize',source=native)
        gpu=_run(compiler,'--allow-unregistered-dialect','--tessera-native-tape-to-gpu=backend='+backend,source=buffered)
        inputs=len(contract['inputs'])
        specs=tuple(TensorSpec(f'arg{i}','fp32',shape,i>=inputs) for i,shape in enumerate(shapes))+(IndexSpec('scratch',1,1),)
        first,rest=gpu.split('\n',1)
        prefix='module attributes {'
        if not first.startswith(prefix) or not first.endswith('} {'):
            raise ValueError('native tape module lacks its product attributes')
        attributes=first[len(prefix):-3]
        gpu=attach_tensor_contract('module {\n'+rest,specs,grid=(1,1,1),block=(1,1,1))
        gpu=gpu.replace(prefix,prefix+attributes+', ',1)
        packages.append(build_native_gpu_storage(gpu,compiler=compiler,llvm_bin=llvm_bin,backend=backend,chip=chip))
    if lineages[0]!=lineages[1]:
        raise ValueError('persistent tape native products have different lineage')
    f,b=contracts
    if b['inputs']!=f['inputs']+f['results'] or b['results']!=f['inputs']:
        raise ValueError('persistent tape requires gradients for every primal input')
    result=PersistentTapePair(packages[0],packages[1],hashlib.sha256(lineages[0].encode()).hexdigest())
    result.validate()
    return result


@dataclass(frozen=True)
class PersistentTapePair:
    forward: NativeGPUStoragePackage
    backward: NativeGPUStoragePackage
    lineage_digest: str

    def validate(self):
        contracts=[]
        for role,package in zip(('forward','backward'),(self.forward,self.backward),strict=True):
            package.validate()
            c=json.loads(_attribute(package.arena_ir,'tessera.autodiff.product_abi'))
            lineage=_attribute(package.arena_ir,'tessera.autodiff.product_pair')
            if c.get('role')!=role or c.get('schema')!=1 or hashlib.sha256(lineage.encode()).hexdigest()!=self.lineage_digest:
                raise ValueError('persistent tape product identity disagrees')
            manifest=read_tensor_contract(package)
            inputs=len(c['inputs'])
            expected=tuple(TensorSpec(f'arg{i}','fp32',_shape(t),i>=inputs)
                           for i,t in enumerate(c['inputs']+c['results']))+(IndexSpec('scratch',1,1),)
            if (tensor_contract_specs(manifest)!=expected or manifest['grid']!=[1,1,1]
                    or manifest['block']!=[1,1,1]):
                raise ValueError('persistent tape tensor binding disagrees with native product ABI')
            contracts.append(c)
        if (self.forward.backend,self.forward.chip)!=(self.backward.backend,self.backward.chip):
            raise ValueError('persistent tape products require the same backend')
        f,b=contracts
        if b['inputs']!=f['inputs']+f['results'] or b['results']!=f['inputs']:
            raise ValueError('persistent tape residual ABI disagrees')
        return f,b

    def capture(self,*inputs):
        return PersistentTapeFrame(self,inputs)


class _ReadOnly:
    def __init__(self,buffer):
        self._buffer=buffer

    @property
    def __cuda_array_interface__(self):
        view=dict(self._buffer.__cuda_array_interface__)
        view['data']=(view['data'][0],True)
        return view


class PersistentTapeFrame:
    def __init__(self,pair,inputs):
        f,b=pair.validate()
        if len(inputs)!=f['primal_inputs']:
            raise ValueError('persistent tape primal arity disagrees')
        self.pair=pair
        self.closed=False
        self.buffers=[]
        self._lock=threading.RLock()
        self._identities=(pair.forward.binding_digest,pair.backward.binding_digest)
        self._bindings=[]
        for package,contract in zip((pair.forward,pair.backward),(f,b),strict=True):
            names=[f'arg{i}' for i in range(len(contract['inputs'])+len(contract['results']))]+['scratch']
            signature=inspect.Signature([inspect.Parameter(n,inspect.Parameter.POSITIONAL_ONLY) for n in names])
            self._bindings.append(generate_tensor_binding(package,signature))
        self._bindings[0]._bound=pair.forward.bind()
        native=self._bindings[0]._bound
        self.check,self.sync=native._check,native._sync
        self._driver=native._driver
        cuda=pair.forward.backend=='nvidia'
        P,S=ct.c_void_p,ct.c_size_t
        def bind(cu,hip,args):
            fn=getattr(self._driver,cu if cuda else hip)
            fn.argtypes,fn.restype=args,ct.c_int
            return fn
        self.alloc=bind('cuMemAlloc_v2','hipMalloc',[ct.POINTER(P),S])
        self.free=bind('cuMemFree_v2','hipFree',[P])
        self.copy=bind('cuMemcpyDtoD_v2','hipMemcpyDtoD',[P,P,S])
        self.context_type=P if cuda else ct.c_int
        self.current=bind('cuCtxGetCurrent','hipGetDevice',[ct.POINTER(self.context_type)])
        self.context=self.context_type()
        self.check(self.current(ct.byref(self.context)))
        try:
            self._inputs=tuple(_Buffer(self,_shape(t)) for t in f['inputs'])
            self._outputs=tuple(_Buffer(self,_shape(t)) for t in f['results'])
            self._bindings[0]._resident(*inputs,*self._outputs,1)
            self.check(self.sync())
            for source,target in zip(inputs,self._inputs,strict=True):
                self.check(self.copy(target.pointer,P(source.__cuda_array_interface__['data'][0]),target.nbytes))
            self._bindings[0](*self._inputs,*self._outputs,1)
            self._primal_count=f['primal_results']
            self.primals=tuple(_ReadOnly(v) for v in self._outputs[:self._primal_count])
            self.residuals=tuple(_ReadOnly(v) for v in self._outputs[self._primal_count:])
        except BaseException:
            self.close()
            raise

    def _ready(self):
        if self.closed:
            raise ValueError('persistent tape frame is closed')
        current=self.context_type()
        self.check(self.current(ct.byref(current)))
        if current.value!=self.context.value:
            raise ValueError('persistent tape requires its owning device context')
        if self._identities!=(self.pair.forward.binding_digest,self.pair.backward.binding_digest):
            raise ValueError('persistent tape package changed')

    def backward(self,*cotangents):
        with self._lock:
            self._ready()
            if len(cotangents)!=self._primal_count:
                raise ValueError('persistent tape cotangent arity disagrees')
            start=len(self.buffers)
            try:
                outputs=tuple(_Buffer(self,v.shape) for v in self._inputs)
                self._bindings[1](*self._inputs,*cotangents,*self._outputs[self._primal_count:],*outputs,1)
                return tuple(_ReadOnly(v) for v in outputs)
            except BaseException:
                self._release(start)
                raise

    def _release(self,start):
        self.check(self.sync())
        while len(self.buffers)>start:
            buffer=self.buffers[-1]
            self.check(self.free(buffer.pointer))
            buffer.pointer=ct.c_void_p()
            self.buffers.pop()

    def close(self):
        with self._lock:
            if self.closed:
                return
            self._ready()
            self._release(0)
            for binding in self._bindings:
                binding.close()
            self.closed=True

    def __enter__(self):
        self._ready()
        return self

    def __exit__(self,*exc):
        self.close()
