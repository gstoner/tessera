"""Independent logical prefixes through native Graph and scale adjoints."""
import itertools,os,subprocess
import pytest
from tests.unit.test_native_scaled_transpose_export import manifest
from tessera.compiler.native_scaled_program import package_native_scaled_vjp

def source(prefixes,ta=False,tb=False,output=(2,3)):
    shapes=((7,3) if ta else (3,7),(5,7) if tb else (7,5),(3,2),(2,2))
    types=["tensor<"+"x".join(map(str,(*prefix,*shape)))+"x"+dtype+">"
           for prefix,shape,dtype in zip(prefixes,shapes,("f8E4M3FN","f8E4M3FN","f32","f32"),strict=True)]
    result="tensor<"+"x".join(map(str,(*output,3,5)))+"xf32>"
    args=", ".join("%"+name+": "+ty for name,ty in zip(("a","b","sa","sb"),types))
    return f"""module attributes {{tessera.target = "rocm", tessera.arch = "gfx1201"}} {{
      func.func @scales({args}) -> {result}
        attributes {{tessera.autodiff = "reverse", tessera.autodiff.wrt_indices = [2,3]}} {{
        %y = tessera.scaled_matmul %a, %b scales(%sa, %sb) {{
          batching = "broadcast", transposeA = {str(ta).lower()}, transposeB = {str(tb).lower()},
          numeric_policy = {{accum = "fp32", execution_mode = "exact_per_block"}},
          scale_layout = {{granularity = "block", block = [3,4], format = "fp32"}}
        }} : ({", ".join(types)}) -> {result}
        return %y : {result}
      }}
    }}"""

def native(text,args=()):
    return subprocess.run([os.environ["TESSERA_OPT"],*args],input=text,text=True,capture_output=True,timeout=180)

@pytest.mark.compiler_route
@pytest.mark.usefixtures("production_compiler")
@pytest.mark.parametrize("mask",range(1,16))
@pytest.mark.parametrize("ta,tb",tuple(itertools.product((False,True),repeat=2)))
def test_native_independent_prefix_and_reverse_export(mask,ta,tb):
    prefixes=tuple((2,3) if mask & (1<<i) else () for i in range(4))
    graph=source(prefixes,ta,tb)
    checked=native(graph)
    assert checked.returncode==0,checked.stderr
    paired=native(graph,("--tessera-autodiff-paired=export-scaled-transpose=true",))
    assert paired.returncode==0,paired.stderr
    program=manifest(paired.stdout)
    assert program["gradient_roles"]==[2,3]
    assert len(program["steps"])==2
    for role,out in zip((2,3),program["outputs"],strict=True):
        assert program["buffers"][out]["shape"]==list((*prefixes[role],*( (3,2) if role==2 else (2,2))))
    assert "tensor.generate" in paired.stdout and "scf.for" in paired.stdout

@pytest.mark.compiler_route
@pytest.mark.usefixtures("production_compiler")
@pytest.mark.parametrize("prefixes,output",[
    (((2,3),(3,2),(),()),(2,3)),
    (((2,1),(3,),(),(1,3)),(6,)),
    (((2,1),(3,),(),(1,3)),(2,1)),
    (((2,1),(3,),(),(1,3)),(1,2,3)),
    (((0,3),(),(),()),(0,3)),
])
def test_native_rejects_incompatible_or_forged_prefix(prefixes,output):
    result=native(source(prefixes,output=output))
    assert result.returncode!=0

@pytest.mark.compiler_route
@pytest.mark.usefixtures("rocm_image_toolchain")
@pytest.mark.parametrize("mask",[1,2,4,8,15])
@pytest.mark.parametrize("ta,tb",[(False,False),(True,True)])
def test_independent_scale_reverse_native_image_package(mask,ta,tb):
    prefixes=tuple((2,3) if mask & (1<<i) else () for i in range(4))
    package=package_native_scaled_vjp(source(prefixes,ta,tb))
    package.validate()
    assert len(package.images)==2
    assert all(image.startswith(b"\x7fELF") for image in package.images)

@pytest.mark.compiler_route
@pytest.mark.usefixtures("rocm_image_toolchain")
def test_singleton_unequal_rank_and_unbatched_reverse_package():
    for prefixes,output in [(((2,1),(3,),(),(1,3)),(2,3)),(((),(),(),()),())]:
        graph=source(prefixes,ta=True,output=output)
        checked=native(graph)
        assert checked.returncode==0,checked.stderr
        package=package_native_scaled_vjp(graph)
        package.validate()
