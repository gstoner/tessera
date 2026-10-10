"""Read compiler-owned actual SM120 tensor members without rebuilding Graph."""
from dataclasses import dataclass
import base64
import json
import re
import struct
from pathlib import Path

from .rocm_native import _tool_digest
from .scheduled_matmul import (find_tessera_opt,run_tessera_opt, ScheduledMatmulArtifact,
                               schedule_split_k)
from .scheduled_kernel import ScheduledKernelArtifact

def _plan(ir):
    matches=re.findall(r'tessera.native.sm120_tensor_program_json = "([^"]+)"',ir)
    if len(matches)!=1:
        raise ValueError("native tensor export requires one program record")
    text=base64.b64decode(matches[0],validate=True).decode()
    if json.loads(text).get("schema") not in {"tessera.native.sm120_tensor_program.v1","tessera.native.sm120_tensor_program.v2","tessera.native.sm120_tensor_program.v3","tessera.native.sm120_tensor_program.v4","tessera.native.sm120_tensor_program.v5","tessera.native.sm120_tensor_program.v6"}:
        raise ValueError("native tensor program schema differs")
    return text

def _one(pattern,text):
    matches=re.findall(pattern,text)
    if len(matches)!=1:
        raise ValueError("native tensor member lost a unique Schedule/Tile field")
    return matches[0]

def _string(name,text,default):
    values=re.findall(r'(?<![\w.])'+re.escape(name)+r' = "([^"]+)"',text)
    if len(values)>1:
        raise ValueError("native tensor member has ambiguous policy")
    return values[0] if values else default

@dataclass(frozen=True)
class NativeSM120TensorGraph:
    source_ir:str
    outlined_ir:str
    plan_json:str
    tool:Path
    compiler_identity:str

    @property
    def manifest(self):
        return json.loads(self.plan_json)

    def project_member(self,index):
        if type(index) is not int or not 0 <= index < len(self.manifest["steps"]):
            raise ValueError("native tensor member index is outside its native program")
        if _tool_digest(self.tool)!=self.compiler_identity:
            raise RuntimeError("native tensor compiler changed after export")
        ir=run_tessera_opt(self.tool,self.source_ir,
            f"--tessera-autodiff-forward=export-scaled-primal=true select-scaled-member={index}")
        if ir!=self.manifest["member_graphs"][index]:
            raise ValueError("native tensor member differs from captured program")
        return ir

    def scheduled_members(self):
        plan=self.manifest
        stages=[]
        for index in range(len(plan["steps"])):
            graph=self.project_member(index)
            schedule=run_tessera_opt(self.tool,graph,"--tessera-graph-to-schedule")
            tile=run_tessera_opt(self.tool,schedule,"--tessera-schedule-to-tile")
            digest=_one(r'tessera.schedule_hash = "([0-9a-f]{64})"',tile)
            entry=_one(r'\bllvm.func @(\w+)\(',tile)
            buffers=plan["buffers"]
            inputs=plan["steps"][index]["inputs"]
            output=buffers[plan["steps"][index]["outputs"][0]]
            lhs=buffers[inputs[0]]
            storage=lhs["storage"]
            dtype={"f16":"fp16","bf16":"bf16"}[storage]
            artifact: ScheduledKernelArtifact | ScheduledMatmulArtifact
            if index<len(plan["steps"])-1:
                operation=plan["steps"][index]["operation"]
                softmax=operation=="tessera.softmax"
                kind="softmax" if softmax else "layernorm" if operation=="tessera.layer_norm" else "rmsnorm"
                mode=_string("schedule",tile,"serial")
                from .nvidia_native import _mlir_f32_bits
                epsilon=0.0 if softmax else struct.unpack("f",
                    _mlir_f32_bits(_one(r'tessera.norm_epsilon = ([^ ]+) : f32',tile)))[0]
                shape=tuple(lhs["shape"])
                artifact=ScheduledKernelArtifact(
                    graph,schedule,tile,"nvidia_sm120","sm_120",entry,
                    "softmax" if softmax else "norm",kind,"source","edge",
                    shape,tuple(output["shape"]),dtype,storage,"f32",
                    -1,False,shape[0],shape[1],1,1,1,
                    int(_one(r'tessera.workgroup_size = (\d+) : i64',tile)),
                    digest,schedule=mode,epsilon=epsilon)
            else:
                rhs=buffers[inputs[1]]
                m,k=lhs["shape"]
                right_k,n=rhs["shape"]
                if right_k!=k or tuple(output["shape"])!=(m,n) or rhs["storage"]!=storage:
                    raise ValueError("native tensor consumer storage/shape differs")
                output_dtype={"f16":"fp16","f32":"fp32"}[output["storage"]]
                decision=_one(r'(?m)^\s*%[^=]+ = (schedule\.matmul[^\n]+)',schedule)
                bias=bool(re.search(r'\bbias = true\b',decision))
                residual=bool(re.search(r'\bresidual = true\b',decision))
                if len(inputs)!=2+int(bias)+int(residual):
                    raise ValueError("native tensor consumer epilogue operands differ")
                split,reduction=schedule_split_k(schedule)
                artifact=ScheduledMatmulArtifact(
                    graph,schedule,tile,"nvidia_sm120","sm_120",entry,
                    "edge","rhs","out",m,n,k,dtype,dtype,output_dtype,storage,"f32",
                    int(_one(r'tessera.macro_tile_m = (\d+) : i64',tile)),
                    int(_one(r'tessera.macro_tile_n = (\d+) : i64',tile)),digest,
                    bias_name="bias" if bias else None,
                    residual_name="residual" if residual else None,
                    activation=_string("activation",decision,"none"),
                    dynamic_m="M" in plan.get("dynamic_axes",[]),
                    dynamic_n="N" in plan.get("dynamic_axes",[]),
                    dynamic_k="K" in plan.get("dynamic_axes",[]),
                    b_layout=_one(r'\bb_layout = "(row_major|col_major)"',schedule),
                    split_k=split,split_k_reduction=reduction)
            artifact.validate()
            stages.append(artifact)
        if _tool_digest(self.tool)!=self.compiler_identity:
            raise RuntimeError("native tensor compiler changed during lowering")
        return tuple(stages)

def export_native_sm120_tensor_graph(source,*,tool=None):
    if not isinstance(source,str):
        raise TypeError("native tensor export consumes frontend-authored MLIR text")
    tool=find_tessera_opt() if tool is None else tool
    if tool is None:
        raise RuntimeError("native tensor export requires a matching compiler")
    identity=_tool_digest(tool)
    ir=run_tessera_opt(tool,source,"--tessera-autodiff-forward=export-scaled-primal=true")
    if _tool_digest(tool)!=identity:
        raise RuntimeError("native tensor compiler changed during export")
    return NativeSM120TensorGraph(source,ir,_plan(ir),tool,identity)


def validate_native_tensor_plan(text):
    """Check portable allocation and SSA lifetime metadata without compiler calls."""
    if not isinstance(text,str):
        raise ValueError("native tensor plan must be serialized JSON")
    plan=json.loads(text)
    keys={"schema","source_graph_ir","root","role_indices","buffers","steps","output","member_graphs"}
    if isinstance(plan,dict) and plan.get("schema") in {"tessera.native.sm120_tensor_program.v2","tessera.native.sm120_tensor_program.v4","tessera.native.sm120_tensor_program.v6"}:
        keys.update({"active_shape","shape_bounds","dynamic_axes","original_graph_ir"})
    if (not isinstance(plan,dict) or set(plan)!=keys or
            plan["schema"] not in {"tessera.native.sm120_tensor_program.v1","tessera.native.sm120_tensor_program.v2","tessera.native.sm120_tensor_program.v3","tessera.native.sm120_tensor_program.v4","tessera.native.sm120_tensor_program.v5","tessera.native.sm120_tensor_program.v6"}):
        raise ValueError("native tensor program schema differs")
    if plan["schema"].endswith((".v2",".v4",".v6")):
        if not isinstance(plan["original_graph_ir"],str) or not plan["original_graph_ir"]:
            raise ValueError("native tensor original Graph witness differs")
        axes=plan["dynamic_axes"]
        if (not isinstance(axes,list) or not axes or
                any(type(axis) is not str or axis not in {"M","N","K"} for axis in axes) or
                axes!=[axis for axis in ("M","N","K") if axis in axes]):
            raise ValueError("native tensor dynamic axes differ")
        active,bounds=plan["active_shape"],plan["shape_bounds"]
        if (not isinstance(active,list) or not isinstance(bounds,list) or len(active)!=3 or len(bounds)!=3 or
                any(type(v) is not int or not 0<v<2**31 for v in [*active,*bounds]) or
                any(value>bound or (axis not in axes and value!=bound) for axis,value,bound in
                    zip(("M","N","K"),active,bounds,strict=True))):
            raise ValueError("native tensor active shape/capacity differs")
    roles=plan["role_indices"]
    if (not isinstance(roles,list) or len(roles) not in (2,3,4) or
            any(type(v) is not int for v in roles) or sorted(roles)!=list(range(len(roles)))):
        raise ValueError("native tensor argument roles differ")
    count=len(roles)
    steps=plan["steps"]
    dag=plan["schema"].endswith((".v5",".v6"))
    chain=plan["schema"].endswith((".v3",".v4",".v5",".v6"))
    if (not isinstance(steps,list) or (not (2 if dag else 3)<=len(steps)<=64 if chain else len(steps)!=2) or type(plan["output"]) is not int or
            plan["output"]!=count+len(steps)-1):
        raise ValueError("native tensor output/step count differs")
    if (not isinstance(plan["root"],str) or not plan["root"] or
            not isinstance(plan["source_graph_ir"],str) or not plan["source_graph_ir"] or
            not isinstance(plan["member_graphs"],list) or len(plan["member_graphs"])!=len(steps) or
            any(not isinstance(v,str) or not v for v in plan["member_graphs"])):
        raise ValueError("native tensor retained Graphs differ")
    for index,step in enumerate(steps):
        expected_inputs=([roles[0]] if index==0 else [count+index-1]) if index<len(steps)-1 else [count+index-1,*roles[1:]]
        if dag:
            inputs=step.get("inputs") if isinstance(step,dict) else None
            arity=1 if index<len(steps)-1 else count
            if (not isinstance(inputs,list) or len(inputs)!=arity or
                    any(type(v) is not int or not 0<=v<count+index for v in inputs) or
                    (index==len(steps)-1 and inputs[2:]!=roles[2:])):
                raise ValueError("native tensor DAG input prefix differs")
            expected_inputs=inputs
        if (not isinstance(step,dict) or set(step)!={"step","operation","member","inputs","outputs"} or
                type(step["step"]) is not int or step["step"]!=index or
                step["member"]!=plan["root"]+"__tensor_member_"+str(index) or
                step["inputs"]!=expected_inputs or step["outputs"]!=[count+index] or
                any(type(v) is not int for v in [*step["inputs"],*step["outputs"]])):
            raise ValueError("native tensor member SSA differs")
        if step["operation"] not in (
                {"tessera.rmsnorm","tessera.layer_norm","tessera.softmax"} if index<len(steps)-1 else {"tessera.matmul"}):
            raise ValueError("native tensor member operation differs")
    buffers=plan["buffers"]
    if not isinstance(buffers,list) or len(buffers)!=count+len(steps):
        raise ValueError("native tensor buffer count differs")
    widths={"f16":2,"bf16":2,"f32":4}
    for index,b in enumerate(buffers):
        if not isinstance(b,dict) or set(b)!={"id","bytes","shape","storage","ownership","first_write","last_read"}:
            raise ValueError("native tensor buffer fields differ")
        shape=b["shape"]
        if (type(b["id"]) is not int or b["id"]!=index or
                not isinstance(shape,list) or len(shape) not in (1,2) or
                any(type(v) is not int or v<=0 for v in shape) or b["storage"] not in widths):
            raise ValueError("native tensor buffer shape/storage differs")
        size=widths[b["storage"]]
        for extent in shape:
            size*=extent
        if type(b["bytes"]) is not int or b["bytes"]!=size or size>2**63-1:
            raise ValueError("native tensor buffer bytes differ")
        write=-1 if index<count else index-count
        read=max([step["step"] for step in steps if index in step["inputs"]]+[write])
        if index==plan["output"]:read=len(steps)
        ownership="readonly_input" if index<count else "returned_output" if index==plan["output"] else "private_scratch"
        if (type(b["first_write"]) is not int or b["first_write"]!=write or
                type(b["last_read"]) is not int or b["last_read"]!=read or b["ownership"]!=ownership):
            raise ValueError("native tensor buffer ownership/lifetime differs")
    source=buffers[roles[0]]
    if dag:
        origins={index:index for index in range(count)}
        dependencies={}
        for step in steps[:-1]:
            input_id=step["inputs"][0];output_id=step["outputs"][0]
            if origins[input_id] not in roles[:2]:
                raise ValueError("native tensor DAG producer captured an epilogue role")
            input_buffer,output_buffer=buffers[input_id],buffers[output_id]
            if (input_buffer["shape"]!=output_buffer["shape"] or
                    input_buffer["storage"]!=output_buffer["storage"] or
                    len(input_buffer["shape"])!=2 or input_buffer["storage"] not in {"f16","bf16"}):
                raise ValueError("native tensor DAG changed producer storage")
            origins[output_id]=origins[input_id];dependencies[output_id]=input_id
        lhs_id,rhs_id=steps[-1]["inputs"][:2]
        if origins[lhs_id]!=roles[0] or origins[rhs_id]!=roles[1] or rhs_id<count:
            raise ValueError("native tensor DAG operand roots differ")
        live=set()
        for value in (lhs_id,rhs_id):
            while value in dependencies:
                live.add(value);value=dependencies[value]
        if live!=set(dependencies):
            raise ValueError("native tensor DAG has an unused producer")
        lhs,rhs,out=buffers[lhs_id],buffers[rhs_id],buffers[plan["output"]]
        if (len(lhs["shape"])!=2 or len(rhs["shape"])!=2 or
                lhs["shape"][1]!=rhs["shape"][0] or lhs["storage"]!=rhs["storage"] or
                out["shape"]!=[lhs["shape"][0],rhs["shape"][1]] or out["storage"] not in {"f16","f32"}):
            raise ValueError("native tensor DAG consumer shape/storage differs")
    else:
        for row in buffers[count:count+len(steps)-1]:
            if row["shape"]!=source["shape"] or row["storage"]!=source["storage"]:
                raise ValueError("native tensor chain changed producer storage")
    if plan["schema"].endswith((".v2",".v4",".v6")):
        m,n,k=plan["shape_bounds"]
        # Optional roles follow the consumer's bias/residual bindings. The
        # portable edge validator checks their named semantics.
        if source["shape"]!=[m,k] or buffers[roles[1]]["shape"]!=[k,n] or buffers[plan["output"]]["shape"]!=[m,n]:
            raise ValueError("native tensor capacity buffers differ")
    return plan
