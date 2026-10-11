"""Shape-polymorphic frontend certificates and bounded SM120 JIT selection."""
from __future__ import annotations
import ast
from collections import OrderedDict
import ctypes as ct
import dis
import inspect
import threading
import textwrap
import types
import weakref


def validate_bounds(bounds):
    if (not isinstance(bounds,dict) or not bounds
            or any(type(axis) is not str or axis not in {"M","N","K"}
                   or type(bound) is not int or not 0<bound<(1<<31) for axis,bound in bounds.items())):
        raise ValueError("shape_bounds requires positive integer M/N/K capacities")
    return tuple((axis,bounds[axis]) for axis in ("M","N","K") if axis in bounds)


def _path(node):
    if isinstance(node,ast.Name):return (node.id,)
    if isinstance(node,ast.Attribute):return _path(node.value)+(node.attr,)
    raise ValueError("bounded LHS requires direct registered operation calls")


class SourceCertificate:
    """No shape/value branches, hidden helpers, arithmetic or mutable keywords."""
    def __init__(self,fn,source):
        import tessera as ts
        self.fn,self.code=fn,fn.__code__
        nodes=ast.parse(textwrap.dedent(source or ""))
        functions=[n for n in nodes.body if isinstance(n,ast.FunctionDef) and n.name==fn.__name__]
        if len(functions)!=1:raise ValueError("bounded LHS requires a source certificate")
        known=set(inspect.signature(fn).parameters)
        paths=[]
        def expression(node):
            if isinstance(node,ast.Name) and node.id in known:return
            if isinstance(node,ast.Constant) and type(node.value) in (str,int,float,bool,type(None)):return
            if isinstance(node,ast.UnaryOp) and isinstance(node.op,(ast.USub,ast.UAdd)) and isinstance(node.operand,ast.Constant):
                expression(node.operand);return
            if isinstance(node,ast.Call):
                paths.append(_path(node.func))
                if any(isinstance(a,ast.Starred) for a in node.args) or any(k.arg is None for k in node.keywords):
                    raise ValueError("bounded LHS requires explicit operation operands")
                for arg in node.args:expression(arg)
                for keyword in node.keywords:expression(keyword.value)
                return
            raise ValueError("bounded LHS requires shape-independent straight-line operations")
        statements=functions[0].body
        if statements and isinstance(statements[0],ast.Expr) and isinstance(statements[0].value,ast.Constant):
            statements=statements[1:]
        for statement in statements:
            if isinstance(statement,ast.Assign) and len(statement.targets)==1 and isinstance(statement.targets[0],ast.Name):
                expression(statement.value);known.add(statement.targets[0].id)
            elif isinstance(statement,ast.Return) and statement is statements[-1]:
                expression(statement.value)
            else:raise ValueError("bounded LHS requires shape-independent straight-line operations")
        allowed_ops=(ts.ops.matmul,ts.ops.gemm,ts.ops.rmsnorm,ts.ops.layer_norm,ts.ops.softmax)
        self.paths=tuple(paths)
        self.calls=tuple(self.resolve(path) for path in paths)
        if not self.calls or any(type(call) not in (types.FunctionType,types.MethodType)
                                 or call not in allowed_ops for call in self.calls):
            raise ValueError("bounded LHS requires registered producer/matmul operations")
        self.call_codes=tuple(getattr(getattr(call,"__func__",call),"__code__",None) for call in self.calls)
        roots={path[0] for path in paths}
        attrs={name for path in paths for name in path[1:]}
        # Validate the live function as well as supplied source text. A clean
        # source string cannot hide shape branches/helpers in actual bytecode.
        allowed={"RESUME","LOAD_GLOBAL","LOAD_DEREF","LOAD_ATTR","LOAD_METHOD","LOAD_CONST",
                 "CALL","CALL_KW","PRECALL","KW_NAMES","PUSH_NULL","RETURN_VALUE","RETURN_CONST",
                 "COPY_FREE_VARS","CACHE","EXTENDED_ARG","NOP"}
        for instruction in dis.get_instructions(fn):
            if (instruction.opname not in allowed and not instruction.opname.startswith(("LOAD_FAST","STORE_FAST"))):
                raise ValueError("bounded LHS live code requires a shape-independent certificate")
            if instruction.opname in {"LOAD_GLOBAL","LOAD_DEREF"} and instruction.argval not in roots:
                raise ValueError("bounded LHS live code has an uncertified dependency")
            if instruction.opname in {"LOAD_ATTR","LOAD_METHOD"} and instruction.argval not in attrs:
                raise ValueError("bounded LHS live code has an uncertified attribute")

    def resolve(self,path):
        closure=inspect.getclosurevars(self.fn)
        scope={**self.fn.__globals__,**closure.nonlocals}
        try:
            value=scope[path[0]]
            for name in path[1:]:value=getattr(value,name)
            return value
        except (KeyError,AttributeError) as exc:
            raise ValueError("bounded LHS operation certificate dependency differs") from exc

    def validate(self):
        if self.fn.__code__ is not self.code:
            raise ValueError("bounded LHS source certificate changed")
        for path,expected,code in zip(self.paths,self.calls,self.call_codes,strict=True):
            actual=self.resolve(path)
            if actual!=expected or getattr(getattr(actual,"__func__",actual),"__code__",None) is not code:
                raise ValueError("bounded LHS operation certificate changed")


_AXES={"source":("M","K"),"rhs":("K","N"),"bias":("N",),"residual":("M","N")}


def specialization_key(roles,ordered,bounds):
    import numpy as np
    resident=any(hasattr(value,"__cuda_array_interface__") for value in ordered)
    if resident:
        if not all(hasattr(value,"__cuda_array_interface__") for value in ordered):
            raise ValueError("bounded resident frontend requires all CUDA roots")
        from .resident_nvidia_tensor import cuda_frontend_specs
        specs=cuda_frontend_specs(ordered)
    else:
        if not all(isinstance(value,np.ndarray) for value in ordered):
            raise ValueError("bounded LHS input rank/type differs")
        specs=tuple((tuple(value.shape),value.dtype) for value in ordered)
    signature=[]
    for role,index in sorted(roles.items()):
        shape,dtype=specs[index]
        axes=_AXES[role]
        if len(shape)!=len(axes):
            raise ValueError("bounded LHS input rank/type differs")
        if any(size<=0 or (axis in bounds and size>bounds[axis])
               for size,axis in zip(shape,axes,strict=True)):
            raise ValueError("bounded LHS active shape is outside its declared bound")
        signature.append((role,index,dtype.name,tuple(bounds.get(axis,size)
            for size,axis in zip(shape,axes,strict=True))))
    source,rhs=specs[roles["source"]][0],specs[roles["rhs"]][0]
    if source[1]!=rhs[0]:
        raise ValueError("bounded LHS source/RHS contraction extents differ")
    m,n=source[0],rhs[1]
    for role,shape in (("bias",(n,)),("residual",(m,n))):
        if role in roles and specs[roles[role]][0]!=shape:
            raise ValueError("bounded LHS epilogue active extents differ")
    return tuple(signature)


def _resident_layout_key(programs,key,resident):
    """Retain a host-created column RHS package alongside a resident row one."""
    previous=programs.get(key)
    if (resident and previous is not None and not previous.rhs_chain
            and previous.edge.consumer.descriptor.provenance["b_layout"]!="row_major"):
        return (key,"resident_row_major")
    return key


class BoundedLhsDispatcher:
    def __init__(self,jit,bounds,certificate,rhs_storage_order=None):
        self.jit=weakref.ref(jit)
        self.bounds=tuple(bounds)
        self.rhs_storage_order=rhs_storage_order
        self.certificate=certificate
        self.lock=threading.RLock()
        self.process=__import__("os").getpid()
        self.roles=None
        self.programs=OrderedDict()
        self.graphs={}

    def __call__(self,args,kwargs):
        import os
        import numpy as np
        from dataclasses import replace
        from . import nvidia_tensor_lhs as lhs
        from tessera import runtime as rt
        if os.getpid()!=self.process:
            raise ValueError("bounded LHS cannot cross fork")
        with self.lock:
            self.certificate.validate()
            jit=self.jit()
            if jit is None:
                raise ValueError("bounded LHS frontend owner has been released")
            bound=jit._nvidia_rhs_call_signature.bind(*args,**kwargs)
            bound.apply_defaults()
            ordered=tuple(bound.arguments[name] for name in jit.arg_names)
            resident=any(hasattr(value,"__cuda_array_interface__") for value in ordered)
            if resident and not all(hasattr(value,"__cuda_array_interface__") for value in ordered):
                raise ValueError("bounded resident frontend requires all CUDA roots")
            if len(ordered) not in {2,3,4} or not all(isinstance(value,np.ndarray) or resident for value in ordered):
                raise ValueError("shape_bounds requires tensor producer/matmul inputs")
            bounds=dict(self.bounds)
            key=specialization_key(self.roles,ordered,bounds) if self.roles is not None else None
            key=_resident_layout_key(self.programs,key,resident)
            program=self.programs.get(key)
            if program is None:
                module=jit._traced_autodiff_module(ordered,{})
                module=lhs.project_rhs_storage(module,ordered,dynamic=True,
                                               rhs_storage_order=self.rhs_storage_order)
                if not lhs.candidate(module):
                    raise ValueError("shape_bounds requires a named normalization/softmax -> matmul Graph")
                program=replace(lhs.package_traced_lhs(module,shape_bounds=bounds),
                                argument_names=tuple(jit.arg_names))
                roles=program.semantics["roles"]
                if self.roles is not None and roles!=self.roles:
                    raise ValueError("bounded LHS frontend role certificate changed")
                key=_resident_layout_key(self.programs,specialization_key(roles,ordered,bounds),resident)
                graph=module if program.rhs_chain else lhs._semantic_graph(
                    program.edge.m,program.edge.k,program.edge.n,
                    program.edge.dtype,program.semantics,tuple(bounds))
                if len(self.programs)>=24:
                    oldest,_=self.programs.popitem(last=False)
                    self.graphs.pop(oldest)
                    for call_key in list(jit._nvidia_lhs_prepared_calls):
                        if isinstance(call_key,tuple) and call_key[:2]==("bounded",oldest):
                            jit._nvidia_lhs_prepared_calls.pop(call_key).close()
                self.roles=dict(roles)
                self.programs[key]=program
                self.graphs[key]=graph
            self.programs.move_to_end(key)
            lib=rt._load_nvidia_ptx_launch()
            prepared=(os.environ.get("TESSERA_NVIDIA_PREPARED_LHS","1").lower() not in {"0","off","false"}
                      and lib is not None and hasattr(lib,"tessera_nvidia_matmul_set_dynamic_axes"))
            call_key=("bounded",key)
            if prepared and lib is not None:
                identity=ct.c_uint64()
                lib.tessera_nvidia_matmul_context_identity.argtypes=[ct.POINTER(ct.c_uint64)]
                lib.tessera_nvidia_matmul_context_identity.restype=ct.c_int
                lib.tessera_nvidia_matmul_last_error.restype=ct.c_char_p
                if lib.tessera_nvidia_matmul_context_identity(ct.byref(identity)):
                    reason=lib.tessera_nvidia_matmul_last_error()
                    raise RuntimeError(reason.decode() if reason else "bounded native context unavailable")
                call_key+= (identity.value,)
            _,receipt=jit._launch_nvidia_lhs_program(program,ordered,call_key,prepared=prepared)
            jit.graph_ir=self.graphs[key]
            jit.frontend_authority="tracer"
            return receipt["output"]
