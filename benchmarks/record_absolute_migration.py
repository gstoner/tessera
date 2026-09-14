"""Owning x86 execution of the native absolute Schedule contract."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from tessera.compiler.graph_ir import GraphIRModule, GraphIRFunction, IRArg, IROp, IRType
from tessera.compiler.scheduled_absolute import lower_absolute, lower_floor, lower_ceil, package_absolute
from tessera.compiler.scheduled_matmul import find_tessera_opt
from tessera.compiler.x86_native import _library_path
from tessera import runtime as rt


def module(shape):
    tensor=IRType('tensor<'+'x'.join(map(str,shape))+'xf32>',tuple(map(str,shape)),'fp32')
    return GraphIRModule(functions=[GraphIRFunction(name='absolute',args=[IRArg('x',tensor)],
        result_types=[tensor],body=[IROp(result='out',op_name='tessera.absolute',operands=['%x'],
        operand_types=[str(tensor)],result_type=str(tensor),kwargs={})],return_values=['%out'])])


def record(operation="absolute"):
    if operation not in {"absolute", "floor", "ceil"}:
        raise ValueError("unsupported unary migration")
    rows=[]
    for shape in ((51,),(3,17),(2,3,17)):
        graph = module(shape)
        graph.functions[0].body[0].op_name = "tessera." + operation
        artifact={"absolute":lower_absolute,"floor":lower_floor,"ceil":lower_ceil}[operation](graph)
        packet=package_absolute(artifact,pipeline_name='tessera-lower-to-x86')
        bits=np.resize(np.array([0,0x80000000,1,0x80000001,0x00800000,0x80800000,
             0x7f800000,0xff800000,0x7fc00001,0xffc00001,0xbf800000,0x3fc00000,0xbfc00000,0x7f7fffff],np.uint32),np.prod(shape))
        values=bits.view(np.float32).reshape(shape); out=np.empty_like(values)
        bound=rt.RuntimeArtifact(metadata={'target':'x86'},native_image=packet.image,
              launch_descriptor=packet.descriptor,tile_ir=packet.tile_ir,target_ir=packet.target_ir)
        result=rt.launch(bound,dict(x=values,out=out,N=values.size))
        if not result.get('ok') or result.get('execution_kind')!='native_cpu':
            raise RuntimeError(str(result))
        if operation == "absolute":
            np.testing.assert_array_equal(out.view(np.uint32).ravel(),bits & np.uint32(0x7fffffff))
        else:
            expected = (np.floor if operation == "floor" else np.ceil)(values)
            finite = ~np.isnan(expected)
            np.testing.assert_array_equal(out[finite].view(np.uint32), expected[finite].view(np.uint32))
            np.testing.assert_array_equal(np.isnan(out), np.isnan(expected))
        rows.append(dict(shape=shape,state='passed',schedule_sha256=hashlib.sha256(artifact.schedule_ir.encode()).hexdigest(),
                         image_digest=packet.image.image_digest,promotion_eligible=False))
    return dict(backend='x86',rows=rows,compiler_sha256=hashlib.sha256(find_tessera_opt().read_bytes()).hexdigest(),
                runtime_sha256=hashlib.sha256(_library_path().read_bytes()).hexdigest(),scope=f'f32 {operation} including ragged, signed-zero, subnormal and exceptional inputs')


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,required=True); p.add_argument('--operation',choices=('absolute','floor','ceil'),default='absolute'); args=p.parse_args()
    args.output.write_text(json.dumps(record(args.operation),indent=2)+'\n')
