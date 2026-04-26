from dataclasses import dataclass
from typing import List, Dict, Any
from . import ast as A

@dataclass
class IRFunction:
    name: str
    params: List[str]
    dims: List[str]
    body: List[Dict[str, Any]]
    ret: str

@dataclass
class IRModule:
    funcs: List[IRFunction]

def ast_to_ir(mod: A.Module) -> IRModule:
    ir_funcs: List[IRFunction] = []
    for fn in mod.funcs:
        body_ops: List[Dict[str, Any]] = []
        ret_name = None
        for s in fn.body:
            if isinstance(s, A.Assign):
                if isinstance(s.op, A.MatMul):
                    body_ops.append({"op":"matmul", "out":s.target, "lhs":s.op.lhs, "rhs":s.op.rhs})
                elif isinstance(s.op, A.Add):
                    body_ops.append({"op":"add", "out":s.target, "lhs":s.op.lhs, "rhs":s.op.rhs})
            elif isinstance(s, A.Return):
                ret_name = s.value
        if ret_name is None:
            raise ValueError(f"Function {fn.name} missing return")
        ir_funcs.append(IRFunction(name=fn.name, params=fn.params, dims=fn.dims, body=body_ops, ret=ret_name))
    return IRModule(funcs=ir_funcs)

def dump_text(ir: IRModule) -> str:
    lines = []
    for f in ir.funcs:
        dims = f"[{','.join(f.dims)}]" if f.dims else ""
        lines.append(f"func @{f.name}({', '.join('%'+p for p in f.params)}) {dims} {{")
        for op in f.body:
            if op["op"] == "matmul":
                lines.append(f"  %{op['out']} = matmul %{op['lhs']}, %{op['rhs']}")
            elif op["op"] == "add":
                lines.append(f"  %{op['out']} = add %{op['lhs']}, %{op['rhs']}")
        lines.append(f"  return %{f.ret}")
        lines.append("}")
    return "\n".join(lines) + "\n"
