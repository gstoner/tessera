"""Owning x86 execution of the native absolute Schedule contract."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from tessera.compiler.graph_ir import GraphIRModule, GraphIRFunction, IRArg, IROp, IRType
from tessera.compiler.scheduled_absolute import lower_absolute, package_absolute
from tessera.compiler.scheduled_matmul import find_tessera_opt
from tessera.compiler.x86_native import _library_path
from tessera import runtime as rt


def module(shape):
    tensor=IRType('tensor<'+'x'.join(map(str,shape))+'xf32>',tuple(map(str,shape)),'fp32')
    return GraphIRModule(functions=[GraphIRFunction(name='absolute',args=[IRArg('x',tensor)],
        result_types=[tensor],body=[IROp(result='out',op_name='tessera.absolute',operands=['%x'],
        operand_types=[str(tensor)],result_type=str(tensor),kwargs={})],return_values=['%out'])])


def record():
    rows=[]
    for shape in ((51,),(3,17),(2,3,17)):
        artifact=lower_absolute(module(shape))
        packet=package_absolute(artifact,pipeline_name='tessera-lower-to-x86')
        bits=np.resize(np.array([0,0x80000000,1,0x80000001,0x00800000,0x80800000,
             0x7f800000,0xff800000,0x7fc00001,0xffc00001,0xbf800000,0x7f7fffff],np.uint32),np.prod(shape))
        values=bits.view(np.float32).reshape(shape); out=np.empty_like(values)
        bound=rt.RuntimeArtifact(metadata={'target':'x86'},native_image=packet.image,
              launch_descriptor=packet.descriptor,tile_ir=packet.tile_ir,target_ir=packet.target_ir)
        result=rt.launch(bound,dict(x=values,out=out,N=values.size))
        if not result.get('ok') or result.get('execution_kind')!='native_cpu':
            raise RuntimeError(str(result))
        np.testing.assert_array_equal(out.view(np.uint32).ravel(),bits & np.uint32(0x7fffffff))
        rows.append(dict(shape=shape,state='passed',schedule_sha256=hashlib.sha256(artifact.schedule_ir.encode()).hexdigest(),
                         image_digest=packet.image.image_digest,promotion_eligible=False))
    return dict(backend='x86',rows=rows,compiler_sha256=hashlib.sha256(find_tessera_opt().read_bytes()).hexdigest(),
                runtime_sha256=hashlib.sha256(_library_path().read_bytes()).hexdigest(),scope='f32 absolute bitwise magnitude including ragged and exceptional inputs')


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,required=True); args=p.parse_args()
    args.output.write_text(json.dumps(record(),indent=2)+'\n')
