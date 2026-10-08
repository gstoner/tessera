"""Independent partial K-group frontend and native compiler contracts."""
import itertools,json
import ml_dtypes,numpy as np,pytest
from tests.unit.test_public_independent_scaled_primal import case
from tessera.compiler.rocm_typed_scaled_native import contract,lower_typed_scaled,package_typed_scaled

def partial_case(mask,ta,tb,encoded,k,jvp=False,seed=1031,m=3,n=5):
    scalar,owner,_,row=case(mask,tb,encoded,ta=ta,jvp=jvp,seed=seed)
    rng=np.random.default_rng(seed);g=(k+31)//32;sn=1 if encoded else 3
    suffix=((k,m) if ta else (m,k),(n,k) if tb else (k,n),(m,g),(g,(n+sn-1)//sn))
    values=[]
    for slot,(prefix,shape) in enumerate(zip(row["prefixes"],suffix,strict=True)):
        dims=(*prefix,*shape)
        if slot<2:value=rng.uniform(-.5,.5,dims).astype(ml_dtypes.float8_e4m3fn)
        elif encoded:value=rng.integers(126,129,dims,dtype=np.uint8)
        else:value=rng.uniform(.3,1.3,dims).astype(np.float32)
        values.append(value)
    if jvp:values.extend(rng.uniform(-.5,.5,x.shape).astype(np.float32) for x in values[2:4])
    return scalar,owner,values,row

def oracle(values,row):
    def primal(v):
        a=v[0].astype(np.float64);b=v[1].astype(np.float64)
        if row["transposeA"]:a=a.swapaxes(-1,-2)
        if row["transposeB"]:b=b.swapaxes(-1,-2)
        sa,sb=v[2:4]
        if row["encoded"]:sa,sb=np.exp2(sa.astype(np.float64)-127),np.exp2(sb.astype(np.float64)-127)
        else:sa,sb=sa.astype(np.float64),sb.astype(np.float64)
        m,k=a.shape[-2:];n=b.shape[-1];sn=1 if row["encoded"] else 3
        out=np.zeros((*row["output_prefix"],m,n),np.float64)
        for g in range((k+31)//32):
            dot=a[..., :,g*32:min(k,(g+1)*32)]@b[...,g*32:min(k,(g+1)*32),:]
            out+=dot*sa[...,g,None]*np.take(sb[...,g,:],np.arange(n)//sn,axis=-1)[...,None,:]
        return out
    out=[primal(values)]
    if row["kind"]=="paired_jvp":
        out.append(primal([*values[:2],values[4],values[3]])+primal([*values[:2],values[2],values[5]]))
    return out

@pytest.mark.parametrize("mask",range(1,16))
@pytest.mark.parametrize("ta,tb,encoded",tuple(itertools.product((False,True),repeat=3)))
def test_partial_public_projection_retains_ceiling_scales(mask,ta,tb,encoded):
    scalar,owner,values,row=partial_case(mask,ta,tb,encoded,37)
    graph=owner._specialized_autodiff_module(values,{})
    shape,_,_,_=contract(graph)
    assert shape.k==37 and shape.groups==2
    assert graph.functions[0].body[0].kwargs["batching"]=="broadcast"
    assert scalar._frontend_batch_axes is None

@pytest.mark.compiler_route
@pytest.mark.usefixtures("rocm_image_toolchain")
@pytest.mark.parametrize("k",[1,15,17,31,33,37,63,65])
@pytest.mark.parametrize("ta,tb,encoded",tuple(itertools.product((False,True),repeat=3)))
def test_native_partial_group_image_and_static_capacity(k,ta,tb,encoded):
    _,owner,values,_=partial_case(4,ta,tb,encoded,k)
    graph=owner._specialized_autodiff_module(values,{})
    lowered=lower_typed_scaled(graph)
    package=package_typed_scaled(graph,lowered,pipeline_name="tessera-lower-to-rocm")
    manifest=package.descriptor.provenance["native_scaled_primal_program"]
    member=json.loads(manifest["members_json"][0])
    assert member["scalars"]==[3,5,k]
    assert member["image_policy"]=="static_independent_prefix_v1"

@pytest.mark.compiler_route
@pytest.mark.usefixtures("rocm_image_toolchain")
@pytest.mark.parametrize("mutation",["scale_groups","policy","lds","k_extent"])
def test_partial_target_rejects_inconsistent_planes(mutation):
    import os,re,subprocess
    _,owner,values,_=partial_case(4,False,False,False,37)
    graph=owner._specialized_autodiff_module(values,{})
    package=package_typed_scaled(graph,lower_typed_scaled(graph),pipeline_name="tessera-lower-to-rocm")
    target=package.target_ir
    if mutation=="scale_groups":
        target,n=re.subn(r"tensor<2x3x3x2xf32>","tensor<2x3x3x1xf32>",target)
    elif mutation=="policy":target,n=re.subn(r'batching = "broadcast"','batching = "independent_rhs"',target)
    elif mutation=="lds":target,n=re.subn(r'staging = "global"','staging = "lds"',target)
    else:target,n=re.subn(r"k = 37 : i64","k = 38 : i64",target)
    assert n>0
    result=subprocess.run([os.environ["TESSERA_OPT"],
        "--pass-pipeline=builtin.module(tessera-rocm-executable{family=matmul input=directive output=binary arch=gfx1201})"],
        input=target,text=True,capture_output=True,timeout=180)
    assert result.returncode!=0


@pytest.mark.compiler_route
@pytest.mark.usefixtures("rocm_image_toolchain")
@pytest.mark.parametrize("k",[1,37])
@pytest.mark.parametrize("ta,tb",tuple(itertools.product((False,True),repeat=2)))
def test_partial_jvp_owns_three_bounded_scaled_products(k,ta,tb):
    from dataclasses import replace
    from tessera.compiler.native_scaled_program import package_native_scaled_jvp
    _,owner,values,_=partial_case(12,ta,tb,False,k,True)
    graph=owner._specialized_autodiff_module(values[:4],{})
    graph=replace(graph,module_attrs={**graph.module_attrs,
        "tessera.target":'"rocm"',"tessera.arch":'"gfx1201"'})
    package=package_native_scaled_jvp(graph.to_mlir(target="rocm_gfx1201"))
    body=json.loads(package.program_json)
    assert len(package.images)==4
    assert len([s for s in body["steps"] if s["operation"]=="tessera.scaled_matmul"])==3
    assert all(json.loads(m)["scalars"]==[3,5,k] for m in package.members_json[:3])
