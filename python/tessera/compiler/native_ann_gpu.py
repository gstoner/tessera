"""Native frozen ANN programs on the bounded serial CUDA/HIP consumer.

This owns an explicit host-buffer bridge and never labels CPU execution as GPU.
The physical schedule is a correctness baseline, not a tuned GEMM candidate.
"""
from dataclasses import dataclass
import ctypes as ct
import hashlib
import inspect
from pathlib import Path
from types import SimpleNamespace
import threading
import numpy as np
from .native_ann import NativeANNPair, _affine, _affine_error_bounds, _exact_output, _exact_error
from .native_gpu_storage import NativeGPUStoragePackage, _run, build_native_gpu_storage
from .native_gpu_tensor import TensorSpec, IndexSpec
from .native_storage_contract import attach_tensor_contract, generate_tensor_binding
from .native_persistent_tape import _attribute


def materialize_native_ann_gpu(pair, *, compiler, llvm_bin, backend, chip, fuse_elementwise=False):
    if type(fuse_elementwise) is not bool:
        raise ValueError('ANN fusion selection must be boolean')
    pair.validate()
    compiler, llvm_bin = Path(compiler), Path(llvm_bin)
    if hashlib.sha256(compiler.read_bytes()).hexdigest() != pair.compiler_digest:
        raise ValueError('ANN physical compiler differs from rewrite owner')
    packages=[]
    for source in (pair.original, pair.transformed):
        gpu=_prepare_ann_gpu_ir(source,compiler,llvm_bin,backend,fuse_elementwise)
        packages.append(build_native_gpu_storage(gpu,compiler=compiler,llvm_bin=llvm_bin,backend=backend,chip=chip))
    result=NativeANNDevicePair(pair,packages[0],packages[1],compiler,llvm_bin)
    result.validate()
    return result


def _prepare_ann_gpu_ir(source,compiler,llvm_bin,backend,fuse_elementwise=False):
    _, shape, layers, _ = _affine(source)
    native=_run(compiler, '--tessera-to-linalg', source=source)
    native=_run(llvm_bin/'mlir-opt','--allow-unregistered-dialect','--convert-elementwise-to-linalg',
                *(['--linalg-fuse-elementwise-ops'] if fuse_elementwise else []),source=native)
    buffered=_run(llvm_bin/'mlir-opt', '--allow-unregistered-dialect',
        '--one-shot-bufferize=bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map',
        '--buffer-results-to-out-params=modify-public-functions hoist-static-allocs',
        '--convert-linalg-to-loops', '--canonicalize', source=native)
    encoded='"'+''.join('\\'+format(b,'02X') for b in source.encode())+'"'
    pipeline='elementwise-fused-v1' if fuse_elementwise else 'serial-v1'
    buffered=buffered.replace('module {', 'module attributes {tessera.ann.pipeline = \"'+pipeline+'\", tessera.ann.source = '+encoded+'} {',1)
    gpu=_run(compiler, '--allow-unregistered-dialect',
        '--tessera-native-tape-to-gpu=backend='+backend, source=buffered)
    first,rest=gpu.split('\n',1)
    attributes=first[len('module attributes {'):-3]
    specs=(TensorSpec('x','fp32',shape,False),
           TensorSpec('out','fp32',(shape[0],layers[-1][0].shape[1]),True),IndexSpec('scratch',1,1))
    gpu=attach_tensor_contract('module {\n'+rest,specs,grid=(1,1,1),block=(1,1,1))
    gpu=gpu.replace('module attributes {','module attributes {'+attributes+', ',1)
    return gpu


@dataclass(frozen=True)
class NativeANNDevicePair:
    logical: NativeANNPair
    original: NativeGPUStoragePackage
    transformed: NativeGPUStoragePackage
    compiler: Path
    llvm_bin: Path

    def validate(self):
        self.logical.validate()
        if hashlib.sha256(self.compiler.read_bytes()).hexdigest()!=self.logical.compiler_digest:
            raise ValueError('ANN replay compiler identity changed')
        for source,package in zip((self.logical.original,self.logical.transformed),
                                  (self.original,self.transformed),strict=True):
            package.validate()
            if hashlib.sha256((self.llvm_bin/'mlir-opt').read_bytes()).hexdigest()!=package.llvm_digest:
                raise ValueError('ANN physical toolchain identity changed')
            pipeline=_attribute(package.arena_ir,'tessera.ann.pipeline')
            if pipeline not in ('serial-v1','elementwise-fused-v1'):
                raise ValueError('ANN physical optimization pipeline is unsupported')
            gpu=_prepare_ann_gpu_ir(source,self.compiler,self.llvm_bin,package.backend,pipeline=='elementwise-fused-v1')
            replay=_run(self.compiler,'--allow-unregistered-dialect','--tessera-tile-buffer-reuse',
                        '--tessera-tile-buffer-arena','--canonicalize',source=gpu)
            if replay!=package.arena_ir:
                raise ValueError('ANN device artifact disagrees with native source replay')
            if _attribute(package.arena_ir,'tessera.ann.source') != source or package.compiler_digest != self.logical.compiler_digest:
                raise ValueError('ANN device package lost native source ownership')
        if (self.original.backend,self.original.chip)!=(self.transformed.backend,self.transformed.chip):
            raise ValueError('ANN device programs require the same target')

    def bind(self, *, input_bound, absolute_budget):
        return BoundNativeANNDevice(self,input_bound,absolute_budget)


class BoundNativeANNDevice:
    """Reusable native packages; measured calls include H2D, dispatch and D2H."""
    def __init__(self,pair,input_bound,absolute_budget):
        pair.validate()
        if type(absolute_budget) not in (int,float) or not np.isfinite(absolute_budget) or absolute_budget<0:
            raise ValueError('ANN budget must be finite and nonnegative')
        self.bounds=_affine_error_bounds(pair.logical,input_bound)
        self.rewrite_admitted=sum(self.bounds)<=absolute_budget
        self.pair,self.input_bound=pair,input_bound
        self.bindings=[]
        self.pointers=[]
        self.closed=False
        self._lock=threading.RLock()
        signature=inspect.Signature([inspect.Parameter(n,inspect.Parameter.POSITIONAL_ONLY) for n in ('x','out','scratch')])
        try:
            for package in (pair.original,pair.transformed):
                binding=generate_tensor_binding(package,signature)
                binding._bound=package.bind()
                self.bindings.append(binding)
            native=self.bindings[0]._bound
            self.native=native
            cuda=pair.original.backend=='nvidia'
            P,S=ct.c_void_p,ct.c_size_t
            def bind(cu,hip,args):
                fn=getattr(native._driver,cu if cuda else hip)
                fn.argtypes,fn.restype=args,ct.c_int
                return fn
            self.allocate=bind('cuMemAlloc_v2','hipMalloc',[ct.POINTER(P),S])
            self.free=bind('cuMemFree_v2','hipFree',[P])
            self.to_device=bind('cuMemcpyHtoD_v2','hipMemcpyHtoD',[P,P,S])
            self.to_host=bind('cuMemcpyDtoH_v2','hipMemcpyDtoH',[P,P,S])
            self.context_type=P if cuda else ct.c_int
            self.current=bind('cuCtxGetCurrent','hipGetDevice',[ct.POINTER(self.context_type)])
            self.context=self.context_type()
            native._check(self.current(ct.byref(self.context)))
            program=_affine(pair.logical.original)
            self.shape=program[1]
            self.output_shape=(self.shape[0],program[2][-1][0].shape[1])
            self.views=[]
            for shape in (self.shape,self.output_shape):
                pointer=P()
                native._check(self.allocate(ct.byref(pointer),int(np.prod(shape))*4))
                self.pointers.append(pointer)
                self.views.append(SimpleNamespace(__cuda_array_interface__=dict(version=3,shape=shape,
                    typestr='<f4',data=(pointer.value,False))))
        except BaseException:
            self.close()
            raise

    def _ready(self):
        if self.closed:
            raise ValueError('ANN device binding is closed')
        current=self.context_type()
        self.native._check(self.current(ct.byref(current)))
        if current.value!=self.context.value:
            raise ValueError('ANN device binding requires its owning context')

    def run(self,value,*,transformed=False):
        with self._lock:
            self._ready()
            if type(transformed) is not bool:
                raise ValueError('ANN variant must be boolean')
            if transformed and not self.rewrite_admitted:
                raise ValueError('ANN rewrite exceeds the analytic budget')
            value=np.array(value,copy=True,order='C')
            if value.dtype!=np.float32 or value.shape!=self.shape or not np.isfinite(value).all() or np.any(np.abs(value.astype(np.float64))>self.input_bound):
                raise ValueError('ANN input violates the admitted domain')
            output=np.empty(self.output_shape,np.float32)
            self.native._check(self.to_device(self.pointers[0],value.ctypes.data,value.nbytes))
            self.bindings[int(transformed)](*self.views,1)
            self.native._check(self.to_host(output.ctypes.data,self.pointers[1],output.nbytes))
            if not np.isfinite(output).all():
                raise ValueError('ANN device produced nonfinite output')
            return output

    def verify(self,samples):
        with self._lock:
            values=[np.array(v,copy=True) for v in samples]
            if not values:
                raise ValueError('ANN device verification requires probes')
            program=_affine(self.pair.logical.original)
            for value in values:
                for i in ((0,1) if self.rewrite_admitted else (0,)):
                    actual=self.run(value,transformed=bool(i))
                    if _exact_error(actual,_exact_output(program,value))>self.bounds[i]:
                        raise ValueError('ANN device violates its independent analytic oracle')
            return True

    def close(self):
        with self._lock:
            if self.closed:
                return
            if self.pointers:
                self._ready()
                self.native._check(self.native._sync())
                while self.pointers:
                    self.native._check(self.free(self.pointers[-1]))
                    self.pointers.pop()
            for binding in self.bindings:
                binding.close()
            self.bindings=[]
            self.closed=True

    def __enter__(self):
        return self

    def __exit__(self,*exc):
        self.close()


def summarize_native_ann_measurements(reports):
    """Fixed nine-run package admission, with no optional early stopping.

    Reports are measurement evidence, not executable authorization. The existing
    production arbiter still needs an owning-target registration before routing.
    """
    import statistics
    from .apple_route_selector import median_speedup_confidence_interval
    if len(reports)!=9:
        raise ValueError('ANN promotion requires exactly nine independent runs')
    identity_keys=('backend','chip','pair','original','transformed','input_bound','absolute_budget',
                   'bounds','timing_domain','sources','recorder_sha256')
    first=reports[0]
    if first['timing_domain']!='warm_package_h2d_dispatch_d2h_host_wall':
        raise ValueError('ANN timing domain disagrees')
    seen=set()
    speedups=[]
    for report in reports:
        if any(report[k]!=first[k] for k in identity_keys) or report.get('numerical_verified') is not True:
            raise ValueError('ANN measurement identity or numerical evidence disagrees')
        if type(report.get('pid')) is not int or report['pid']<=0 or report['pid'] in seen:
            raise ValueError('ANN measurement requires distinct process runs')
        seen.add(report['pid'])
        samples=report['samples_ms']
        if len(samples)!=2 or any(len(row)!=31 or any(type(v) not in (int,float) or not np.isfinite(v) or v<=0 for v in row) for row in samples):
            raise ValueError('ANN timing evidence must contain positive finite samples')
        medians=[statistics.median(row) for row in samples]
        ratio=medians[0]/medians[1]
        stored=report['medians_ms']
        if (type(stored) is not list or len(stored)!=2 or any(type(v) not in (int,float) or not np.isfinite(v) or v<=0 for v in stored)
                or stored!=medians or type(report['speedup']) not in (int,float) or report['speedup']!=ratio):
            raise ValueError('ANN summary disagrees with raw measurements')
        speedups.append(ratio)
    interval=median_speedup_confidence_interval(speedups)
    assert interval is not None
    return dict(schema=1,backend=first['backend'],chip=first['chip'],pair=first['pair'],
                timing_domain=first['timing_domain'],run_speedups=speedups,
                median_speedup=statistics.median(speedups),median_bounds=list(interval),
                threshold=1.02,performance_eligible=interval[0]>1.02,
                production_promoted=False,
                reason='requires native GPU arbiter integration' if interval[0]>1.02 else 'lower bound does not clear 2 percent margin')


# Separate internal family: CPU ANN verification must not overwrite this GPU
# verifier when both backends register in one process.
from .emit.candidate import Candidate, Tier  # noqa: E402
from .native_ann import ANNRegion  # noqa: E402
ANN_GPU = 'ann_affine_gpu'


@dataclass(frozen=True)
class ANNDeviceRegion:
    logical: ANNRegion
    original_digest: str
    transformed_digest: str
    backend: str
    chip: str

    @property
    def digest(self):
        import json
        return hashlib.sha256(json.dumps((self.logical.digest,self.original_digest,
            self.transformed_digest,self.backend,self.chip)).encode()).hexdigest()


class NativeANNDeviceCandidate(Candidate):
    op=ANN_GPU
    tier=Tier.SYNTHESIZED

    def __init__(self,registration,transformed):
        self.registration=registration
        self.transformed=transformed
        self.target=registration.region.backend
        self.name=('ann_gpu_rewrite_' if transformed else 'ann_gpu_original_')+registration.region.digest

    def available(self):
        return not self.registration.closed and not self.registration.runner.closed

    def applies_to(self,region):
        return (self.available() and isinstance(region,ANNDeviceRegion) and
                region.digest==self.registration.region.digest and
                (not self.transformed or self.registration.runner.rewrite_admitted))

    def applies_to_inputs(self,region,*inputs):
        if not self.applies_to(region) or len(inputs)!=1 or not isinstance(inputs[0],np.ndarray):
            return False
        value=inputs[0]
        return (value.dtype==np.float32 and value.shape==self.registration.runner.shape and
                bool(np.isfinite(value).all()) and
                bool(np.all(np.abs(value.astype(np.float64))<=region.logical.input_bound)))

    def run(self,region,*inputs):
        if not self.applies_to_inputs(region,*inputs):
            raise ValueError('ANN GPU candidate does not admit this artifact, domain or budget')
        return self.registration.runner.run(inputs[0],transformed=self.transformed),'native_gpu'


def _verify_gpu_ann(candidate,region,*,atol,seed):
    if not isinstance(candidate,NativeANNDeviceCandidate) or not candidate.applies_to(region):
        return False
    program=_affine(region.logical.pair.original)
    bound=candidate.registration.runner.bounds[int(candidate.transformed)]
    for value in region.logical.arrays():
        output,tag=candidate.run(region,value)
        if tag!='native_gpu' or _exact_error(output,_exact_output(program,value))>bound:
            return False
    return True


class NativeANNDeviceRegistration:
    """Scoped GPU arbiter candidates with one reusable native binding.

    Original is registered first at equal tier. A declined rewrite budget must
    never remove the unchanged incumbent. A forced or measured selection still
    passes normal arbiter verification and per-invocation input guards.
    """
    def __init__(self,pair,samples,*,input_bound,absolute_budget):
        from .emit.candidate import register_candidate,register_op_kind
        shape=_affine(pair.logical.original)[1]
        values=[np.array(v,copy=True,order='C') for v in samples]
        if not values or any(v.dtype!=np.float32 or v.shape!=shape for v in values):
            raise ValueError('ANN GPU registration requires fp32 probes of the native input shape')
        logical=ANNRegion(pair.logical,input_bound,absolute_budget,tuple(v.tobytes() for v in values))
        self.region=ANNDeviceRegion(logical,pair.original.binding_digest,pair.transformed.binding_digest,
                                    pair.original.backend,pair.original.chip)
        self.runner=pair.bind(input_bound=input_bound,absolute_budget=absolute_budget)
        self.closed=False
        self.candidates=(NativeANNDeviceCandidate(self,False),NativeANNDeviceCandidate(self,True))
        register_op_kind(ANN_GPU,_verify_gpu_ann)
        for candidate in self.candidates:register_candidate(candidate)

    def close(self):
        from .emit.candidate import unregister_candidate
        if self.closed:return
        # A failed completion keeps the owner retryable and candidates guarded
        # by their native binding; never discard retained device allocations.
        self.runner.close()
        for candidate in self.candidates:unregister_candidate(candidate)
        self.closed=True

    def __enter__(self):
        return self

    def __exit__(self,*exc):
        self.close()
