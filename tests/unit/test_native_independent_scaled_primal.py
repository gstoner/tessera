"""Native independent-prefix primal Schedule/Tile image and ABI contracts."""
import itertools
import json
import os
import subprocess
import pytest
from tessera.compiler.native_scaled_program import package_native_scaled_primal

def source(prefixes, output=(2,3), tb=False, encoded=False):
    suffix=((3,64),(5,64) if tb else (64,5),(3,2),(2,5) if encoded else (2,2))
    dtype=("f8E4M3FN","f8E4M3FN","ui8" if encoded else "f32","ui8" if encoded else "f32")
    types=["tensor<"+"x".join(map(str,(*prefix,*shape)))+"x"+dt+">"
           for prefix,shape,dt in zip(prefixes,suffix,dtype,strict=True)]
    result="tensor<"+"x".join(map(str,(*output,3,5)))+"xf32>"
    args=", ".join("%"+name+": "+ty for name,ty in zip(("a","b","sa","sb"),types))
    return f"""module attributes {{tessera.target = "rocm", tessera.arch = "gfx1201"}} {{
      func.func @independent({args}) -> {result} {{
        %y = tessera.scaled_matmul %a, %b scales(%sa, %sb) {{
          batching = "broadcast", transposeB = {str(tb).lower()},
          numeric_policy = {{accum = "fp32", execution_mode = "exact_per_block"}},
          scale_layout = {{granularity = "block", block = [{1 if encoded else 3},32],
                           format = "{'e8m0' if encoded else 'fp32'}"}}
        }} : ({", ".join(types)}) -> {result}
        return %y : {result}
      }}
    }}"""

@pytest.mark.parametrize("mask",range(1,16))
@pytest.mark.parametrize("tb,encoded",tuple(itertools.product((False,True),repeat=2)))
def test_native_independent_primal_package(mask,tb,encoded):
    prefixes=tuple((2,3) if mask&(1<<slot) else () for slot in range(4))
    package=package_native_scaled_primal(source(prefixes,tb=tb,encoded=encoded),project_image_identity=False)
    program=json.loads(package.program_json)
    member=json.loads(package.members_json[0])
    assert program["steps"][0]["batching"]=="broadcast"
    assert program["buffers"][4]["shape"]==[2,3,3,5]
    assert member["scalars"]==[3,5,64]
    assert member["geometry"][2]==6
    package.validate()
    assert package.images[0].startswith(b"\x7fELF")

@pytest.mark.parametrize("prefixes,output",[
    (((2,1),(3,),(),(1,3)),(2,3)),
    (((),(),(),()),()),
    (((2,1,3),(),(1,1,3),(2,1,1)),(2,1,3)),
])
def test_singleton_unequal_rank_and_unbatched_primal(prefixes,output):
    package=package_native_scaled_primal(source(prefixes,output=output),project_image_identity=False)
    assert json.loads(package.members_json[0])["geometry"][2]==__import__("math").prod(output)
    package.validate()

def test_independent_prefix_changes_sealed_schedule_identity():
    texts=[source(((2,3),(),(),()),tb=True),source(((),(),(2,3),()),tb=True)]
    hashes=[]
    for text in texts:
        result=subprocess.run([os.environ["TESSERA_OPT"],"--tessera-graph-to-schedule"],
                              input=text,text=True,capture_output=True,timeout=180)
        assert result.returncode==0,result.stderr
        import re
        hashes.append(re.findall(r'artifact_hash = "([^"]+)"',result.stdout))
    assert hashes[0] and hashes[1] and hashes[0]!=hashes[1]

@pytest.mark.parametrize("tb,encoded",tuple(itertools.product((False,True),repeat=2)))
def test_default_image_policy_retains_independent_plane_types(tb,encoded):
    text=source(((),(),(2,3),()),tb=tb,encoded=encoded)
    default=package_native_scaled_primal(text)
    static=package_native_scaled_primal(text,project_image_identity=False)
    assert default.images==static.images
    assert json.loads(default.members_json[0])["image_policy"]=="static_independent_prefix_v1"

def test_native_scale_jvp_independent_primal_members():
    from tessera.compiler.native_scaled_program import package_native_scaled_jvp
    text=source(((),(),(2,3),()),tb=True)
    text=text.replace("-> tensor<2x3x3x5xf32> {",
        '-> tensor<2x3x3x5xf32> attributes {tessera.autodiff = "forward", tessera.autodiff.wrt_indices = [2,3]} {',1)
    package=package_native_scaled_jvp(text)
    package.validate()
    assert len(package.images)==4
    assert json.loads(package.program_json)["kind"]=="paired_jvp"

@pytest.fixture(scope="module")
def independent_target():
    text=source(((),(),(2,3),()),tb=True)
    pipeline="builtin.module(tessera-graph-to-schedule,tessera-schedule-to-tile,tessera-rocm-executable{family=matmul input=tile output=target arch=gfx1201})"
    result=subprocess.run([os.environ["TESSERA_OPT"],"--pass-pipeline="+pipeline],
                          input=text,text=True,capture_output=True,timeout=180)
    assert result.returncode==0,result.stderr
    return result.stdout

@pytest.mark.parametrize("mutation",["count","output_prefix","scale_suffix","scale_storage","policy"])
def test_native_target_rejects_forged_independent_planes(independent_target,mutation):
    import re
    target=independent_target
    if mutation=="policy":
        target,n=re.subn(r'batching = "broadcast"','batching = "independent_rhs"',target)
    elif mutation=="count":
        target,n=re.subn(r"batch_count = 6 : i64","batch_count = 5 : i64",target)
    elif mutation=="output_prefix":
        target,n=re.subn(r"batch_result = tensor<2x3x3x5xf32>",
                        "batch_result = tensor<3x2x3x5xf32>",target)
    else:
        match=re.search(r"batch_operands = (\[[^\]]+\])",target)
        assert match
        types=match.group(1)
        old="tensor<2x3x3x2xf32>"
        assert old in types
        types=types.replace(old,"tensor<2x3x4x2xf32>" if mutation=="scale_suffix"
                            else "tensor<2x3x3x2xui8>")
        target=target[:match.start(1)]+types+target[match.end(1):]
        n=1
    assert n==1
    pipeline="builtin.module(tessera-rocm-executable{family=matmul input=directive output=binary arch=gfx1201})"
    result=subprocess.run([os.environ["TESSERA_OPT"],"--pass-pipeline="+pipeline],
                          input=target,text=True,capture_output=True,timeout=180)
    assert result.returncode!=0
    assert "independent scaled batch metadata differs" in result.stderr
