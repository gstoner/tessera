"""Native ownership of the verified named producer -> SM120 matmul edge."""
from __future__ import annotations
import copy
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .nvidia_tensor_lhs import TracedLhsProgram
import ctypes as ct
import os
import struct
import numpy as np
from .prepared_nvidia_matmul import PreparedMatmulCall


class PreparedLhsCall(PreparedMatmulCall):
    def __init__(self, program):
        from .native_artifact import LaunchGeometry, OrderingSemantics, WorkspaceRequirement
        from .nvidia_tensor_lhs import _semantic_graph, _partitions
        from .nvidia_tensor_rhs import NvidiaNormRhsProgram
        program.validate()
        edge = program.edge
        producer, consumer = edge.producer, edge.consumer
        descriptor = consumer.descriptor
        provenance = descriptor.provenance
        dynamic=bool(edge.dynamic_m or edge.dynamic_k or edge.dynamic_n)
        macro = (not dynamic and provenance.get("physical_route") ==
                 "macro_cta_cp_async_2stage_shared_ab_" + provenance.get("storage", "")
                 and provenance.get("storage") in {"f16", "bf16"}
                 and provenance.get("b_layout") == "col_major"
                 and descriptor.entry_symbol.endswith("_macro_kernel"))
        policy = ("sm120_scheduled_macro_cta_32x32_mn" if macro
                  else "sm120_scheduled_typed_16x8_mn")
        if ((not macro and provenance.get("physical_route") != "typed_fragment_global")
                or bool(provenance.get("dynamic_shape_bounds") is not None) != dynamic
                or descriptor.geometry != LaunchGeometry(policy=policy)
                or descriptor.workspace != WorkspaceRequirement()
                or descriptor.dynamic_local_memory_bytes or descriptor.dynamic_local_memory_expression
                or descriptor.ordering != OrderingSemantics(
                    ordered_submission=True, residency="none", synchronization=("completion",))):
            raise ValueError("prepared tensor edge requires synchronous typed matmul")
        pd = producer.descriptor
        pp = pd.provenance
        norm = pp["kind"] in {"rmsnorm", "layernorm"}
        policy = ("sm120_norm_" + pp["schedule"] + "_rows" if norm
                  else ("sm120_softmax_cooperative_128_rows" if pp["schedule"] == "cooperative_128"
                        else "sm120_softmax_thread_per_row_128"))
        scalar_names = ("Rows", "Columns") if norm else ("Rows", "K")
        if (pp["schedule"] not in {"serial", "cooperative_128"}
                or pd.geometry != LaunchGeometry(policy=policy)
                or pd.workspace != WorkspaceRequirement()
                or pd.dynamic_local_memory_bytes or pd.dynamic_local_memory_expression
                or pd.ordering != descriptor.ordering
                or tuple(s.name for s in sorted(pd.scalars, key=lambda s:s.ordinal)) != scalar_names
                or any(s.dtype != "int64" for s in pd.scalars)):
            raise ValueError("prepared tensor producer geometry/scalar ABI differs")
        artifact = NvidiaNormRhsProgram.runtime_artifact(consumer)
        if program.native_plan_json is not None:
            names=tuple(item.name for item in sorted(descriptor.buffers,key=lambda b:b.ordinal)
                        if item.direction=="input")
            self._initialize(artifact,None,program.graph_ir,dynamic=dynamic,binding_names=names)
        else:
            graph = _semantic_graph(edge.m, edge.k, edge.n, edge.dtype, program.semantics)
            _, consumer_graph = _partitions(graph)
            self._initialize(artifact,consumer_graph,program.graph_ir,dynamic=dynamic)
        try:
            if not hasattr(self.lib, "tessera_nvidia_matmul_attach_producer"):
                raise ValueError("prepared tensor edge runtime unavailable")
            producers=program.producer_chain or (producer,)
            for index,stage in enumerate(producers):
                attach=(self.lib.tessera_nvidia_matmul_attach_producer if index==0
                        else self.lib.tessera_nvidia_matmul_append_producer)
                attach.argtypes = [ct.c_uint64, ct.c_void_p, ct.c_size_t, ct.c_char_p, ct.c_int]
                attach.restype = ct.c_int
                image = ct.create_string_buffer(stage.image.payload)
                self._check(attach(self.handle,image,len(stage.image.payload),
                    stage.descriptor.entry_symbol.encode(),
                    int(stage.descriptor.provenance["schedule"]=="cooperative_128")))
            if program.rhs_chain:
                if not hasattr(self.lib,"tessera_nvidia_matmul_attach_rhs_producer"):
                    raise ValueError("native two-sided tensor runtime unavailable")
                attach_rhs=self.lib.tessera_nvidia_matmul_attach_rhs_producer
                attach_rhs.argtypes=[ct.c_uint64,ct.c_void_p,ct.c_size_t,ct.c_char_p,ct.c_int,ct.c_int]
                attach_rhs.restype=ct.c_int
                for index,stage in enumerate(program.rhs_chain):
                    image=ct.create_string_buffer(stage.image.payload)
                    self._check(attach_rhs(self.handle,image,len(stage.image.payload),
                        stage.descriptor.entry_symbol.encode(),
                        int(stage.descriptor.provenance["schedule"]=="cooperative_128"),int(index>0)))
            roles = program.semantics["roles"]
            self.input_positions = tuple(roles[role] for role in (
                ["source", "rhs"] + (["bias"] if "bias" in roles else [])
                + (["residual"] if "residual" in roles else [])))
            self.rhs_layout = provenance["b_layout"]
            self.program = program
            self.resident_snapshot = copy.deepcopy(program)
            self.semantics_snapshot = copy.deepcopy(program.semantics)
            self.producer_snapshot = copy.deepcopy(pd)
            self.chain_snapshot = copy.deepcopy(program.producer_chain)
            self.rhs_snapshot = copy.deepcopy(program.rhs_chain)
            self.graph_snapshot = program.graph_ir
            binding=("prepared_cpp_dynamic_tensor_matmul" if dynamic else "prepared_cpp_tensor_matmul")
            self.component_receipts = tuple(dict(
                ok=True, execution_kind="native_gpu", runtime_status="executed",
                compiler_path="canonical_scheduled_tile_consumer",
                native_call_binding=binding,
                image_digest=p.image.image_digest,
                launch_descriptor_digest=p.descriptor.descriptor_digest,
                artifact_hash=NvidiaNormRhsProgram.runtime_artifact(p).artifact_hash)
                for p in (*producers,*program.rhs_chain,consumer))
            self.receipt_fields.update(compiler_path="canonical_nvidia_lhs_program",
                                       native_call_binding=binding)
        except Exception:
            self.close()
            raise

    def __call__(self, ordered):
        if (self.program.semantics != self.semantics_snapshot
                or self.program.edge.producer.descriptor != self.producer_snapshot
                or self.program.graph_ir != self.graph_snapshot
                or self.program.producer_chain != self.chain_snapshot
                or self.program.rhs_chain != self.rhs_snapshot):
            raise ValueError("prepared tensor edge semantic/producer contract changed")
        values = list(ordered)
        # Host views are packed to the sealed device storage contract. No
        # Python arithmetic or intermediate tensor participates in execution.
        for ordinal, position in enumerate(self.input_positions):
            value = values[position]
            if not isinstance(value, np.ndarray):
                raise TypeError("prepared tensor edge expects host arrays")
            order = "F" if ordinal == 1 and self.rhs_layout == "col_major" else "C"
            values[position] = np.asarray(value, order=order)
        output, receipt = super().__call__(values)
        self._profile_shape=output.shape
        receipt["component_receipts"] = tuple(dict(r) for r in self.component_receipts)
        return output, receipt


    def _resident_call(self,ordered,output,*,stream,repeats=0):
        from .nvidia_tensor_dag import _checked_device_arguments
        from .resident_nvidia_tensor import ordered_resident_views
        from .prepared_nvidia_matmul import HostView
        if self.pid != os.getpid() or not self._finalizer.alive:
            raise ValueError("prepared resident tensor owner is closed or belongs to another process")
        if not self.program.rhs_chain or self.program!=self.resident_snapshot:
            raise ValueError("prepared resident DAG package changed or has no RHS chain")
        self.program.validate()
        roots,_=_checked_device_arguments(self.program,list(ordered))
        values=[*roots,output]
        views,streams=ordered_resident_views(values,stream,writable_from=len(roots))
        declared=(ct.c_uint64*len(streams))(*streams)
        arguments=(self.handle,views,len(values),declared,len(streams),ct.c_void_p(stream))
        common=[ct.c_uint64,ct.POINTER(HostView),ct.c_size_t,
                ct.POINTER(ct.c_uint64),ct.c_size_t,ct.c_void_p]
        if repeats:
            name="tessera_nvidia_matmul_profile_dag_resident_ordered"
            if not hasattr(self.lib,name):raise RuntimeError("native ordered resident profiler unavailable")
            fn=getattr(self.lib,name)
            fn.argtypes=common+[ct.c_int,ct.POINTER(ct.c_float),ct.c_size_t,ct.POINTER(ct.c_float)]
            fn.restype=ct.c_int
            stages=(ct.c_float*len(self.component_receipts))();program=ct.c_float()
            self._check(fn(*arguments,repeats,stages,len(stages),ct.byref(program)))
            return {"program_ms":program.value,"grouped_stage_ms":list(stages)}
        name="tessera_nvidia_matmul_invoke_dag_resident_ordered"
        if not hasattr(self.lib,name):raise RuntimeError("native ordered resident DAG API unavailable")
        fn=getattr(self.lib,name);fn.argtypes=common;fn.restype=ct.c_int
        self._check(fn(*arguments))
        return {"component_receipts":tuple({**receipt,
            "native_call_binding":"prepared_cpp_ordered_resident_tensor_dag"}
            for receipt in self.component_receipts)}

    def resident_to_host(self,ordered):
        """Native owner completes borrowed reads and copies one independent result."""
        from .nvidia_tensor_dag import _checked_device_arguments
        from .resident_nvidia_tensor import ordered_resident_views
        from .prepared_nvidia_matmul import HostView
        if self.pid != os.getpid() or not self._finalizer.alive:
            raise ValueError("prepared resident tensor owner is closed or belongs to another process")
        if not self.program.rhs_chain or self.program!=self.resident_snapshot:
            raise ValueError("prepared resident DAG package changed or has no RHS chain")
        # Constructor admission and the exact deep snapshot above seal all
        # images/contracts. Repeating full IR validation adds host work without
        # strengthening this unchanged native owner.
        roots,shape=_checked_device_arguments(self.program,list(ordered))
        views,streams=ordered_resident_views(roots,None,writable_from=len(roots))
        declared=(ct.c_uint64*len(streams))(*streams)
        output=np.empty(shape,self.output_dtype)
        destination=HostView()
        destination.data,destination.bytes,destination.rank=output.ctypes.data,output.nbytes,2
        destination.dtype=2 if output.dtype==np.float16 else 1
        destination.shape[:]=output.shape;destination.strides[:]=output.strides
        name="tessera_nvidia_matmul_invoke_dag_resident_to_host_ordered"
        if not hasattr(self.lib,name):
            raise RuntimeError("native ordered resident completed-output API unavailable")
        fn=getattr(self.lib,name)
        fn.argtypes=[ct.c_uint64,ct.POINTER(HostView),ct.c_size_t,
                     ct.POINTER(ct.c_uint64),ct.c_size_t,ct.POINTER(HostView)]
        fn.restype=ct.c_int
        self._check(fn(self.handle,views,len(roots),declared,len(streams),ct.byref(destination)))
        receipts=tuple({**receipt,"native_call_binding":"prepared_cpp_ordered_resident_tensor_dag"}
                       for receipt in self.component_receipts)
        return output,{**self.receipt_fields,"component_receipts":receipts,"output":output,
                       "native_call_binding":"prepared_cpp_ordered_resident_tensor_dag"}

    def invoke_resident(self,ordered,output,*,stream):
        """Borrow declared CUDA roots through native synchronous completion."""
        return self._resident_call(ordered,output,stream=stream)

    def profile_resident(self,ordered,output,*,stream,repeats=128):
        """Profile borrowed roots without host copies or retained raw pointers."""
        if type(repeats) is not int or not 1<=repeats<=1000000:
            raise ValueError("resident profile requires a positive bounded repeat count")
        return self._resident_call(ordered,output,stream=stream,repeats=repeats)

    def profile(self, repeats=128):
        """Profile the last host frame; native lease checks reject other owners."""
        if self.pid != os.getpid() or not self._finalizer.alive:
            raise ValueError("prepared tensor owner is closed or belongs to another process")
        if type(repeats) is not int or not 1<=repeats<=1000000 or not hasattr(self,"_profile_shape"):
            raise ValueError("profile requires a successful host frame and positive repeat count")
        self.program.validate()
        from .prepared_nvidia_matmul import HostView
        output=np.empty(self._profile_shape,self.output_dtype)
        view=HostView()
        view.data,view.bytes,view.rank=output.ctypes.data,output.nbytes,2
        view.dtype=2 if output.dtype==np.float16 else 1
        view.shape[:]=output.shape;view.strides[:]=output.strides
        count=len(self.component_receipts)
        stages=(ct.c_float*count)();program=ct.c_float()
        fn=self.lib.tessera_nvidia_matmul_profile
        fn.argtypes=[ct.c_uint64,ct.c_int,ct.POINTER(ct.c_float),ct.c_size_t,
                     ct.POINTER(ct.c_float),ct.POINTER(HostView)]
        fn.restype=ct.c_int
        self._check(fn(self.handle,repeats,stages,count,ct.byref(program),ct.byref(view)))
        return {"program_ms":program.value,"grouped_stage_ms":list(stages),"output":output}

    def scratch_stats(self):
        if self.pid != os.getpid() or not self._finalizer.alive:
            raise ValueError("prepared tensor edge is closed or belongs to another process")
        fn = self.lib.tessera_nvidia_matmul_scratch_stats
        fn.argtypes = [ct.c_uint64,ct.POINTER(ct.c_size_t),ct.POINTER(ct.c_size_t)]
        fn.restype = ct.c_int
        capacity, allocations = ct.c_size_t(), ct.c_size_t()
        self._check(fn(self.handle,ct.byref(capacity),ct.byref(allocations)))
        return capacity.value, allocations.value


# A portable manifest never contains CUDA handles. Each process admits its
# CPU certificate once, then owns a bounded set of context-specific handles.
from collections import OrderedDict
import inspect
import threading

_portable_process = os.getpid()
_portable_lock = threading.RLock()
_portable_programs: OrderedDict[str | None, tuple[object, TracedLhsProgram]] = OrderedDict()
_portable_owners: OrderedDict[tuple[str | None, int], PreparedLhsCall] = OrderedDict()
_PORTABLE_LIMIT = 24


def _same_json(left, right):
    """Type-strict equality: True must never alias the integer 1 in a seal."""
    if type(left) is not type(right):
        return False
    if isinstance(left,dict):
        return (left.keys() == right.keys()
                and all(type(key) is str and _same_json(value,right[key])
                        for key,value in left.items()))
    if type(left) is list:
        return len(left) == len(right) and all(
            _same_json(a,b) for a,b in zip(left,right,strict=True))
    if type(left) is float:
        return struct.pack(">d",left) == struct.pack(">d",right)
    return type(left) in (str,int,bool,type(None)) and left == right


def _portable_pid_guard():
    if os.getpid() != _portable_process:
        raise ValueError("portable native tensor owners cannot cross fork")


def clear_portable_lhs_owners():
    """Retire native owners before clearing their admitted CPU certificates."""
    _portable_pid_guard()
    with _portable_lock:
        for call in _portable_owners.values():
            call.close()
        _portable_owners.clear()
        _portable_programs.clear()


def _portable_arrays(program, args):
    from collections.abc import Mapping
    edge = program.edge
    signature = inspect.Signature([inspect.Parameter(
        name,inspect.Parameter.POSITIONAL_OR_KEYWORD) for name in program.argument_names])
    if isinstance(args, Mapping):
        bound = signature.bind(**args)
    elif isinstance(args,(tuple,list)):
        bound = signature.bind(*args)
    else:
        raise TypeError("native program requires positional or named inputs")
    ordered = [np.asarray(bound.arguments[name]) for name in program.argument_names]
    for role,index in program.semantics["roles"].items():
        package = edge.producer if role == "source" else edge.consumer
        name = (edge.producer_input_name if role == "source" else
                edge.consumer_rhs_name if role == "rhs" else role if program.native_plan_json is not None else "arg" + str(index))
        binding = edge._binding(package,name,"input")
        shape,dynamic = edge._shape_bound(package,name,binding.rank)
        if (ordered[index].ndim != binding.rank or any(
                actual<=0 or actual>bound or (not varying and actual!=bound)
                for actual,bound,varying in zip(ordered[index].shape,shape,dynamic,strict=True))
                or ordered[index].dtype.name != {
                "fp16":"float16","bf16":"bfloat16","fp32":"float32"}[binding.dtype]):
            raise ValueError(f"{role} must match the sealed {binding.dtype} shape bound {shape}")
    return ordered


def launch_portable_lhs(artifact, args):
    """Return a native receipt, or None when the canonical control is selected."""
    from tessera import runtime as rt
    mode = os.environ.get("TESSERA_NVIDIA_PREPARED_LHS_REPLAY",
                          os.environ.get("TESSERA_NVIDIA_PREPARED_LHS","1"))
    if mode.lower() in {"0","off","false"}:
        return None
    _portable_pid_guard()
    lib = rt._load_nvidia_ptx_launch()
    if lib is None or not hasattr(lib,"tessera_nvidia_matmul_context_identity"):
        return None
    metadata = artifact.metadata or {}
    data = metadata.get("native_program")
    digest = data.get("contract_digest") if isinstance(data,dict) else None
    with _portable_lock:
        record = _portable_programs.get(digest) if isinstance(digest,str) else None
        if record is None or not _same_json(record[0],data):
            from .nvidia_tensor_lhs import from_manifest
            program = from_manifest(data)
            snapshot = copy.deepcopy(data)
            # A changed certificate must pass full admission even if its caller
            # reused a display/digest key. Never reinterpret an existing owner.
            if record is not None:
                for key in list(_portable_owners):
                    if key[0] == digest:
                        _portable_owners.pop(key).close()
            if len(_portable_programs) >= _PORTABLE_LIMIT:
                oldest,_ = _portable_programs.popitem(last=False)
                for key in list(_portable_owners):
                    if key[0] == oldest:
                        _portable_owners.pop(key).close()
            _portable_programs[digest] = (snapshot,program)
        else:
            program = record[1]
        _portable_programs.move_to_end(digest)
        if (metadata.get("target") != "nvidia_sm120"
                or artifact.graph_ir != program.graph_ir
                or metadata.get("arg_names") != list(program.argument_names)):
            raise rt.ArtifactContractError("E_LAUNCH_BINDING_MISMATCH",
                                          "native program parent Graph/argument ABI mismatch")
        ordered = _portable_arrays(program,args)
        identity = ct.c_uint64()
        fn = lib.tessera_nvidia_matmul_context_identity
        fn.argtypes = [ct.POINTER(ct.c_uint64)]
        fn.restype = ct.c_int
        last_error = lib.tessera_nvidia_matmul_last_error
        last_error.argtypes = []
        last_error.restype = ct.c_char_p
        if fn(ct.byref(identity)):
            reason = last_error()
            raise RuntimeError(reason.decode() if reason else "native context unavailable")
        key = (digest,identity.value)
        call = _portable_owners.get(key)
        if call is None or not call._finalizer.alive:
            call = PreparedLhsCall(program)
            if len(_portable_owners) >= _PORTABLE_LIMIT:
                _,old = _portable_owners.popitem(last=False)
                old.close()
            _portable_owners[key] = call
        _portable_owners.move_to_end(key)
        return call(ordered)[1]


def execute_chain_resident(program, values):
    """Upload frontend inputs once; native ownership executes the entire chain."""
    program.validate()
    roles = program.semantics["roles"]
    return program.edge.execute_resident(
        values[roles["source"]], values[roles["rhs"]],
        bias=values[roles["bias"]] if "bias" in roles else None,
        residual=values[roles["residual"]] if "residual" in roles else None,
        _producer_chain=program.producer_chain)
