"""Native optional ReLU mapping must survive verified Schedule/Tile projection."""
import pytest
from tessera.compiler.scheduled_matmul import find_tessera_opt, run_tessera_opt

def graph(storage, rows=257, columns=257, mode="flat_128"):
    return f'''module attributes {{tessera.target = "nvidia_sm120", tessera.arch = "sm_120"}} {{
  func.func @relu(%x: tensor<{rows}x{columns}x{storage}>) -> tensor<{rows}x{columns}x{storage}> {{
    %out = "tessera.relu"(%x) {{schedule = "{mode}"}} : (tensor<{rows}x{columns}x{storage}>) -> tensor<{rows}x{columns}x{storage}>
    return %out : tensor<{rows}x{columns}x{storage}>
  }}
}}'''

@pytest.mark.compiler_route
@pytest.mark.parametrize("storage", ["f16", "bf16"])
def test_flat_relu_policy_reaches_native_tile(storage):
    tool=find_tessera_opt()
    if tool is None: pytest.skip("requires production compiler")
    schedule=run_tessera_opt(tool,graph(storage),"--tessera-graph-to-schedule")
    assert 'schedule = "flat_128"' in schedule
    tile=run_tessera_opt(tool,schedule,"--tessera-schedule-to-tile")
    assert "tile.activation_kernel" in tile
    assert 'schedule = "flat_128"' in tile
    assert "_flat_128_" in tile
    assert "tessera.relu" not in tile
    with pytest.raises(RuntimeError,match="altered|decision|hash"):
        run_tessera_opt(tool,schedule.replace('schedule = "flat_128"','schedule = "serial"'),"--tessera-schedule-to-tile")

@pytest.mark.compiler_route
def test_flat_relu_capacity_overflow_is_refused_by_native_schedule():
    tool=find_tessera_opt()
    if tool is None: pytest.skip("requires production compiler")
    with pytest.raises(RuntimeError):
        run_tessera_opt(tool,graph("f16",2**30,3),"--tessera-graph-to-schedule")

@pytest.fixture
def sm120():
    from tessera import runtime as rt
    if find_tessera_opt() is None or rt._nvidia_device_name() != "sm_120":
        pytest.skip("requires owning SM120 device and production compiler")
    return rt

def activation_package(storage, shape, mode):
    from tessera.compiler.native_sm120_tensor_program import export_native_sm120_tensor_graph
    from tessera.compiler.nvidia_native import package_scheduled_kernel
    m,k=shape
    source=f'''module attributes {{tessera.target = "nvidia_sm120", tessera.arch = "sm_120"}} {{
func.func @edge(%x: tensor<{m}x{k}x{storage}>, %w: tensor<{k}x1x{storage}>) -> tensor<{m}x1xf32> {{
%edge = "tessera.relu"(%x) {{schedule = "{mode}"}} : (tensor<{m}x{k}x{storage}>) -> tensor<{m}x{k}x{storage}>
%out = "tessera.matmul"(%edge, %w) {{rhs_storage_order = "row_major"}} : (tensor<{m}x{k}x{storage}>, tensor<{k}x1x{storage}>) -> tensor<{m}x1xf32>
return %out : tensor<{m}x1xf32>
}}
}}'''
    native=export_native_sm120_tensor_graph(source)
    artifact=native.scheduled_members()[0]
    return artifact, package_scheduled_kernel(artifact,pipeline_name="tessera-nvidia-pipeline-sm120")

@pytest.mark.parametrize("storage",["f16","bf16"])
@pytest.mark.parametrize("shape",[(256,256),(257,257),(65537,1)])
def test_flat_relu_exhaustive_storage_bits(sm120,storage,shape):
    import numpy as np
    import ml_dtypes
    artifact,package=activation_package(storage,shape,"flat_128")
    dtype=np.float16 if storage=="f16" else ml_dtypes.bfloat16
    bits=np.resize(np.arange(65536,dtype=np.uint16),shape)
    exponent,fraction=(0x7c00,0x3ff) if storage=="f16" else (0x7f80,0x7f)
    is_nan=((bits & exponent)==exponent) & ((bits & fraction)!=0)
    replace=((bits & 0x8000)!=0) & ~is_nan
    if storage=="bf16": replace &= (bits & 0x7fff)!=0
    expected=bits.copy()
    expected[replace]=0
    source=bits.view(dtype)
    output=np.full(shape,0x5a5a,dtype=np.uint16).view(dtype)
    sm120._submit_nvidia_sm120_native(package.image,package.descriptor,
        {artifact.input_name:source,artifact.output_name:output},
        {"Rows":shape[0],"Columns":shape[1]},None)
    np.testing.assert_array_equal(output.view(np.uint16),expected)
    np.testing.assert_array_equal(source.view(np.uint16),bits)

@pytest.mark.parametrize("storage",["f16","bf16"])
def test_flat_composed_program_preserves_images_numerics_and_ownership(sm120,storage):
    import numpy as np
    import ml_dtypes
    from tests.unit.test_sm120_scheduled_activation import _frontend_program
    from tessera.compiler.nvidia_tensor_lhs import package_traced_lhs
    from tessera.compiler.prepared_nvidia_lhs import PreparedLhsCall
    dtype=np.float16 if storage=="f16" else ml_dtypes.bfloat16
    _,(a,b),public=_frontend_program(dtype,(129,256,33))
    module,_=public._trace_frontend_capture((a,b),{})
    before=module.to_mlir(target="nvidia_sm120",canonical=True)
    serial=package_traced_lhs(module,activation_schedule="serial")
    flat=package_traced_lhs(module,activation_schedule="flat_128")
    assert module.to_mlir(target="nvidia_sm120",canonical=True)==before
    assert serial.producer_chain[0].image.payload==flat.producer_chain[0].image.payload
    assert serial.edge.consumer.image.payload==flat.edge.consumer.image.payload
    assert flat.producer_chain[-1].descriptor.geometry.policy=="sm120_activation_flat_elements"
    original_a,original_b=a.copy(),b.copy()
    owner=PreparedLhsCall(flat)
    retained=[]
    try:
        for factor in (1.0,0.5):
            rhs=(b.astype(np.float32)*factor).astype(dtype)
            wide=a.astype(np.float32)
            normalized=(wide/np.sqrt(np.mean(wide*wide,axis=-1,keepdims=True)+np.float32(1e-5))).astype(dtype)
            oracle=np.maximum(0,normalized).astype(np.float32) @ rhs.astype(np.float32)
            output,receipt=owner((a,rhs))
            assert receipt["execution_kind"]=="native_gpu"
            assert len(receipt["component_receipts"])==3
            np.testing.assert_allclose(output,oracle,rtol=0.02 if storage=="bf16" else 0.003,atol=0.02 if storage=="bf16" else 0.003)
            retained.append((output,output.copy()))
        for output,snapshot in retained: np.testing.assert_array_equal(output,snapshot)
        np.testing.assert_array_equal(a,original_a)
        np.testing.assert_array_equal(b,original_b)
    finally:
        owner.close()

@pytest.mark.parametrize("storage",["f16","bf16"])
def test_flat_rejects_old_provider_before_owner_initialization(sm120,storage,monkeypatch):
    import numpy as np
    import ml_dtypes
    from tessera import runtime as rt
    from tests.unit.test_sm120_scheduled_activation import _frontend_program
    from tessera.compiler.nvidia_tensor_lhs import package_traced_lhs
    from tessera.compiler.prepared_nvidia_lhs import PreparedLhsCall
    dtype=np.float16 if storage=="f16" else ml_dtypes.bfloat16
    _,values,public=_frontend_program(dtype,(7,32,9))
    graph,_=public._trace_frontend_capture(values,{})
    program=package_traced_lhs(graph,activation_schedule="flat_128")
    monkeypatch.setattr(rt,"_load_nvidia_ptx_launch",lambda:object())
    def forbidden(*args,**kwargs):
        raise AssertionError("owner initialization must follow capability negotiation")
    monkeypatch.setattr(PreparedLhsCall,"_initialize",forbidden)
    with pytest.raises(ValueError,match="updated native prepared provider"):
        PreparedLhsCall(program)

@pytest.mark.parametrize("storage",["f16","bf16"])
@pytest.mark.parametrize("bounded",[False,True])
def test_flat_two_sided_dag_and_bounded_replay(sm120,storage,bounded):
    import numpy as np
    import ml_dtypes
    import tessera as ts
    from tessera.compiler.nvidia_tensor_lhs import package_traced_lhs
    from tessera.compiler.prepared_nvidia_lhs import PreparedLhsCall
    dtype=np.float16 if storage=="f16" else ml_dtypes.bfloat16
    public=ts.from_text("""def both(a,b):
    x = tessera.ops.relu(a)
    y = tessera.ops.relu(b)
    return tessera.ops.matmul(x,y,output_dtype="fp32")
""",target="nvidia_sm120")
    rng=np.random.default_rng(5101501)
    def values(shape):
        m,k,n=shape
        return (rng.uniform(-1,1,(m,k)).astype(dtype),
                rng.uniform(-.25,.25,(k,n)).astype(dtype))
    inputs=values((17,64,19))
    graph,_=public._trace_frontend_capture(inputs,{})
    options={"shape_bounds":{"M":33,"K":96,"N":35}} if bounded else {}
    program=package_traced_lhs(graph,activation_schedule="flat_128",**options)
    assert program.rhs_chain
    assert all(p.descriptor.provenance["schedule"]=="flat_128"
               for p in (*program.producer_chain,*program.rhs_chain))
    owner=PreparedLhsCall(program)
    retained=[]
    try:
        shapes=((17,64,19),(33,96,35),(7,32,9),(17,64,19)) if bounded else ((17,64,19),)*2
        for shape in shapes:
            a,b=values(shape)
            output,receipt=owner((a,b))
            assert receipt["execution_kind"]=="native_gpu"
            expected=np.maximum(0,a).astype(np.float32) @ np.maximum(0,b).astype(np.float32)
            np.testing.assert_allclose(output,expected,atol=.02 if storage=="bf16" else .003,rtol=.02 if storage=="bf16" else .003)
            retained.append((output,output.copy()))
        for output,snapshot in retained:np.testing.assert_array_equal(output,snapshot)
    finally:owner.close()
