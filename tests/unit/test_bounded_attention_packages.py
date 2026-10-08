"""Bounded packages retain native capacities, identities and shape roles."""
from dataclasses import replace
import pytest
from tessera.compiler import scheduled_checkpoint as checkpoint,nvidia_native as native
from tessera.compiler.resident_attention import checkpoint_shapes
from tessera.compiler.attention_shape_contract import descriptor_attention_shapes
from tests.unit.test_bounded_attention_checkpoint import graph,lower
from tessera.compiler.scheduled_matmul import find_tessera_opt
pytestmark=pytest.mark.skipif(find_tessera_opt() is None,reason="native compiler required")

def scheduled(backward=False,bias=False,*,seeded=False,compact=False):
    source=graph(backward,bias=bias)
    if seeded or compact:
        import json,re
        # Canonical physical names are sealed by the existing compact ABI.
        source=source.replace('"do"','"dO"').replace('"o"','"output"')
        if backward and seeded:
            count=6+int(bias);lse="tensor<1x2x?xf32>"
            lines=source.splitlines()
            for i,line in enumerate(lines):
                if "func.func @checkpoint(" in line:
                    lines[i]=line.replace(") ->",f", %arg{count}: {lse}) ->",1)
                elif "tessera.argument_bindings" in line:
                    found=re.search(r"tessera.argument_bindings = (\[[^\]]*\])",line)
                    names=json.loads(found[1])+["row_seed"]
                    lines[i]=line[:found.start(1)]+json.dumps(names)+line[found.end(1):]
                elif '"tessera_attn.checkpoint_backward"' in line:
                    lines[i]=line.replace(") {",f", %arg{count}) {{",1).replace("causal = true","causal = true, lse_cotangent = true")
                elif line.lstrip().startswith(": ("):
                    lines[i]=line.replace(") ->",f", {lse}) ->",1)
            source="\n".join(lines)+"\n"
        if backward and compact:
            source=source.replace("tessera.result_bindings =",
                'tessera.checkpoint_gradient_activity = array<i64: 1, 0, 1>, tessera.checkpoint_gradient_output = "compact_v1", tessera.checkpoint_gradient_launch = "packed_v1", tessera.checkpoint_gradient_threads = 128 : i64, tessera.result_bindings =')
    schedule=lower(source,"--tessera-graph-to-schedule")
    tile=lower(schedule,"--tessera-schedule-to-tile")
    return checkpoint._decode_checkpoint(source,schedule,tile,backward=backward,compact_gradients=compact and backward)

def packages(monkeypatch,bias=False):
    monkeypatch.setattr(native,"_compile_tile_ir",lambda text,entry:(text,"// PTX",{},"compiler","toolchain",(),"cold"))
    return native.package_scheduled_checkpoint_pair(scheduled(bias=bias),scheduled(True,bias),pipeline_name="tessera-nvidia-pipeline-sm120")

@pytest.mark.parametrize("bias",[False,True])
def test_bounded_artifact_pair_derives_actual_private_shapes(monkeypatch,bias):
    pair=packages(monkeypatch,bias)
    assert checkpoint_shapes(pair)[0]==(1,2,1,9,11,8,6)
    for dims in ((1,2,1,1,1,8,6),(1,2,1,9,11,8,6),(1,2,1,3,4,8,6)):
        actual,shapes=checkpoint_shapes(pair,runtime_dims=dims)
        assert actual==dims and shapes[0]==(1,2,dims[3],8)
        descriptor_attention_shapes(pair.forward.descriptor,dims)
    assert pair.forward.descriptor.provenance["shape"]==[1,2,1,-(1<<63),-(1<<63),8,6]

@pytest.mark.parametrize("dims",[(1,2,1,10,1,8,6),(1,2,1,1,12,8,6),(2,2,1,3,4,8,6),(1,2,1,0,1,8,6),(1,2,1,True,1,8,6)])
def test_bounded_private_frame_rejects_bad_actual_before_driver(monkeypatch,dims):
    pair=packages(monkeypatch)
    with pytest.raises(ValueError):checkpoint_shapes(pair,runtime_dims=dims)

@pytest.mark.parametrize("mutation",["bounds","guard","symbolic","identity"])
def test_bounded_pair_rejects_metadata_drift(monkeypatch,mutation):
    pair=packages(monkeypatch)
    desc=pair.backward.descriptor;p=dict(desc.provenance)
    if mutation=="guard":desc=replace(desc,shape_guards=desc.shape_guards[:-1])
    elif mutation=="bounds":p["shape_bounds"]=[1,2,1,10,11,8,6]
    elif mutation=="symbolic":p["shape"]=[1,2,1,3,-1,8,6]
    else:p["checkpoint_contract"]="0"*64
    if mutation!="guard":desc=replace(desc,provenance=p)
    changed=replace(pair,backward=replace(pair.backward,descriptor=desc))
    with pytest.raises(ValueError):checkpoint_shapes(changed)

@pytest.mark.parametrize("field,value",[("shape_bounds",()),("dims",(1,2,1,9,11,8,6)),("shape_bounds",(1,2,1,10,11,8,6))])
def test_bounded_artifact_rejects_unsealed_projection_before_compile(monkeypatch,field,value):
    artifact=scheduled()
    monkeypatch.setattr(native,"_compile_tile_ir",lambda *a:pytest.fail("compiled altered bounds"))
    with pytest.raises(ValueError):native.package_scheduled_checkpoint(replace(artifact,**{field:value}),pipeline_name="tessera-nvidia-pipeline-sm120")

@pytest.mark.parametrize("bias",[False,True])
@pytest.mark.parametrize("seeded",[False,True])
@pytest.mark.parametrize("compact",[False,True])
def test_bounded_seeded_and_compact_contracts_bind_actual_shapes(monkeypatch,bias,seeded,compact):
    monkeypatch.setattr(native,"_compile_tile_ir",lambda text,entry:(text,"// PTX",{},"compiler","toolchain",(),"cold"))
    forward=scheduled(bias=bias,seeded=seeded,compact=compact)
    backward=scheduled(True,bias,seeded=seeded,compact=compact)
    pair=native.package_scheduled_checkpoint_pair(forward,backward,pipeline_name="tessera-nvidia-pipeline-sm120")
    actual=(1,2,1,3,4,8,6)
    assert checkpoint_shapes(pair,runtime_dims=actual)[0]==actual
    if seeded:
        from tessera.compiler.lse_cotangent_contract import lse_cotangent_contract
        assert lse_cotangent_contract(pair.backward.descriptor,runtime_dims=actual)[0]==actual
    elif compact:
        from tessera.compiler.compact_attention_contract import compact_attention_contract
        assert compact_attention_contract(pair.backward.descriptor,runtime_dims=actual)[0]==actual

@pytest.mark.parametrize("mutation",["capacity","roles","guard"])
def test_bounded_invocation_rejects_invalid_envelope_before_cuda(monkeypatch,mutation):
    import numpy as np
    from tessera import runtime as rt
    from tessera.compiler.attention_shape_contract import attention_buffer_shapes
    pair=packages(monkeypatch)
    dims=(1,2,1,10 if mutation=="capacity" else 3,4,8,6)
    descriptor=pair.forward.descriptor
    shapes=attention_buffer_shapes(dims)
    buffers={binding.name:np.zeros(shape,np.float32) for binding,shape in zip(descriptor.buffers,shapes,strict=True)}
    scalars=dict(zip(("B","Hq","Hkv","Sq","Sk","D","Dv"),dims,strict=True))
    if mutation=="roles":buffers[descriptor.buffers[2].name]=np.zeros((1,1,5,6),np.float32)
    if mutation=="guard":descriptor=replace(descriptor,shape_guards=descriptor.shape_guards[:-1])
    monkeypatch.setattr(rt,"_load_nvidia_ptx_launch",lambda:pytest.fail("loaded CUDA for invalid bounds"))
    with pytest.raises((ValueError,RuntimeError)):
        rt._submit_nvidia_sm120_native(pair.forward.image,descriptor,buffers,scalars,None)
