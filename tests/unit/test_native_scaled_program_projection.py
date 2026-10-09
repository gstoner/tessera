"""Native compiler/package contracts, including adversarial ABI mutations."""
from pathlib import Path
from dataclasses import replace
import json
import os
import pytest
from tessera.compiler.native_scaled_program import package_native_scaled_jvp

@pytest.fixture(scope="module")
def package():
    if not os.environ.get("TESSERA_OPT"):
        pytest.skip("matching native compiler required")
    source = Path(__file__).resolve().parents[1] / "tessera-ir/phase_f4/autodiff_forward_scaled_matmul_member.mlir"
    return package_native_scaled_jvp(source.read_text())

def test_native_program_has_compiler_owned_sum_and_graph_witness(package):
    package.validate()
    p = json.loads(package.program_json)
    assert "tessera.scaled_matmul" in p["root_ir"]
    assert "tessera.add" in p["root_ir"]
    assert len(p["outputs"]) == 2
    assert p["steps"][-1]["operation"] == "tessera.add"
    m = json.loads(package.members_json[-1])
    assert m["entry"].startswith("tessera_tile_rocm_math_add_")
    assert m["scalars"] == [323]
    assert m["inputs"] == p["steps"][-1]["inputs"]
    assert all(i.startswith(b"\x7fELF") for i in package.images)

@pytest.mark.parametrize("mutation", ["architecture","scalar","future","geometry","witness","output"])
def test_native_program_rejects_member_contract_corruption(package, mutation):
    rows = list(package.members_json)
    member = json.loads(rows[0])
    if mutation == "architecture": member["architecture"] = "gfx1151"
    if mutation == "scalar": member["scalars"][0] += 16
    if mutation == "future": member["inputs"][0] = member["output"]
    if mutation == "geometry": member["geometry"][3:] = [2147483647]*3
    if mutation == "witness": member["program_base64"] = "e30="
    if mutation == "output": member["output"] += 1
    rows[0] = json.dumps(member)
    with pytest.raises(ValueError):
        replace(package, members_json=tuple(rows)).validate()

def test_native_program_rejects_non_elf_member(package):
    images = list(package.images);images[0] = b"not an image"
    with pytest.raises(ValueError):
        replace(package, images=tuple(images)).validate()

def test_native_program_rejects_sibling_target_without_compilation():
    with pytest.raises(ValueError, match="gfx1201"):
        package_native_scaled_jvp("unparsed", target="rocm_gfx1151")

@pytest.fixture(scope="module")
def primal_package():
    if not os.environ.get("TESSERA_OPT"):
        pytest.skip("matching native compiler required")
    from tessera.compiler.native_scaled_program import package_native_scaled_primal
    source=Path(__file__).resolve().parents[1]/"tessera-ir/phase2/e2e_mxfp8_scale_schedule_unsigned.mlir"
    return package_native_scaled_primal(source.read_text())

def test_primal_program_owns_one_output_and_encoded_scale_bytes(primal_package):
    p=json.loads(primal_package.program_json)
    assert p["kind"]=="primal"
    assert p["outputs"]==[4]
    assert len(p["steps"])==1 and p["steps"][0]["inputs"]==[0,1,2,3]
    assert [b["storage"] for b in p["buffers"]]==["f8E4M3FN","f8E4M3FN","ui8","ui8","f32"]
    assert p["buffers"][4]["ownership"]==2
    assert p["buffers"][4]["last_read"]==1
    assert json.loads(primal_package.members_json[0])["scalars"]==[17,19,64]
    from tessera.compiler.native_scaled_program import NativeScaledProgram
    assert NativeScaledProgram.from_manifest(primal_package.to_manifest())==primal_package

def test_paired_output_contract_cannot_be_relabeled_primal(package):
    p=json.loads(package.program_json);p["kind"]="primal"
    with pytest.raises(ValueError,match="ownership"):
        replace(package,program_json=json.dumps(p)).validate()

def test_projected_primal_image_reuses_bytes_without_reusing_launch_extents(primal_package):
    from tessera.compiler.native_scaled_program import package_native_scaled_primal
    source=Path(__file__).resolve().parents[1]/"tessera-ir/phase2/e2e_mxfp8_scale_schedule_unsigned.mlir"
    widened=package_native_scaled_primal(source.read_text().replace("17x","18x"))
    assert widened.images==primal_package.images
    assert json.loads(widened.members_json[0])["entry"]==json.loads(primal_package.members_json[0])["entry"]
    assert json.loads(widened.members_json[0])["scalars"]==[18,19,64]
    assert json.loads(widened.program_json)["buffers"][4]["shape"]==[18,19]
    assert widened.program_json!=primal_package.program_json

def test_native_primal_policy_keeps_fp8_kn_static():
    from tessera.compiler.native_scaled_program import package_native_scaled_primal
    if not os.environ.get("TESSERA_OPT"):pytest.skip("matching native compiler required")
    source=Path(__file__).resolve().parents[1]/"tessera-ir/phase2/e2e_mxfp8_scale_schedule_unsigned.mlir"
    text=source.read_text().replace("xui8","xf32").replace('"e8m0"','"fp32"')
    policy=package_native_scaled_primal(text)
    static=package_native_scaled_primal(text,project_image_identity=False)
    assert json.loads(policy.members_json[0])["image_policy"]=="static_fp8_kn_v1"
    assert policy.images==static.images
    assert json.loads(policy.members_json[0])["scalars"]==[17,19,64]

@pytest.fixture(scope="module")
def shared_primal_package():
    if not os.environ.get("TESSERA_OPT"):
        pytest.skip("matching native compiler required")
    from tessera.compiler.native_scaled_program import package_native_scaled_primal
    source=Path(__file__).resolve().parents[1]/"tessera-ir/phase2/e2e_mxfp8_shared_rhs_batch.mlir"
    return package_native_scaled_primal(source.read_text())

def test_native_batch_program_preserves_logical_shape_and_physical_rows(shared_primal_package):
    p=json.loads(shared_primal_package.program_json)
    assert p["steps"][0]["batching"]=="shared_rhs_rows"
    assert p["buffers"][0]["shape"]==[3,7,64]
    assert p["buffers"][2]["shape"]==[3,7,2]
    assert p["buffers"][4]["shape"]==[3,7,19]
    assert json.loads(shared_primal_package.members_json[0])["scalars"]==[21,19,64]
    shared_primal_package.validate()

@pytest.mark.parametrize("mutation",["output_prefix","scale_prefix","batch_policy"])
def test_native_batch_rejects_consistently_reencoded_bad_logical_contract(shared_primal_package,mutation):
    import base64
    p=json.loads(shared_primal_package.program_json)
    if mutation=="output_prefix":p["buffers"][4]["shape"]=[1,21,19]
    if mutation=="scale_prefix":p["buffers"][2]["shape"]=[1,21,2]
    if mutation=="batch_policy":p["steps"][0]["batching"]="independent_rhs"
    encoded=json.dumps(p)
    member=json.loads(shared_primal_package.members_json[0])
    member["program_base64"]=base64.b64encode(encoded.encode()).decode()
    with pytest.raises(ValueError,match="batch"):
        replace(shared_primal_package,program_json=encoded,members_json=(json.dumps(member),)).validate()

@pytest.fixture(scope="module")
def independent_primal_package():
    if not os.environ.get("TESSERA_OPT"):
        pytest.skip("matching native compiler required")
    from tessera.compiler.native_scaled_program import package_native_scaled_primal
    source=Path(__file__).resolve().parents[1]/"tessera-ir/phase2/e2e_mxfp8_independent_rhs_batch.mlir"
    return package_native_scaled_primal(source.read_text())

def test_independent_batch_native_package_roundtrip(independent_primal_package):
    from tessera.compiler.native_scaled_program import NativeScaledProgram
    p=json.loads(independent_primal_package.program_json)
    m=json.loads(independent_primal_package.members_json[0])
    assert p["steps"][0]["batching"]=="independent_rhs"
    assert p["buffers"][1]["shape"]==[3,64,19]
    assert p["buffers"][3]["shape"]==[3,2,19]
    assert m["scalars"]==[7,19,64] and m["geometry"][2]==3
    assert NativeScaledProgram.from_manifest(independent_primal_package.to_manifest())==independent_primal_package

@pytest.mark.parametrize("mutation",["planes","rhs_prefix","scale_prefix","output_prefix"])
def test_independent_batch_rejects_reencoded_wrong_capacity(independent_primal_package,mutation):
    import base64
    p=json.loads(independent_primal_package.program_json)
    m=json.loads(independent_primal_package.members_json[0])
    if mutation=="planes":m["geometry"][2]=2
    if mutation=="rhs_prefix":p["buffers"][1]["shape"]=[1,192,19]
    if mutation=="scale_prefix":p["buffers"][3]["shape"]=[1,6,19]
    if mutation=="output_prefix":p["buffers"][4]["shape"]=[1,21,19]
    encoded=json.dumps(p)
    m["program_base64"]=base64.b64encode(encoded.encode()).decode()
    with pytest.raises(ValueError,match="batch"):
        replace(independent_primal_package,program_json=encoded,members_json=(json.dumps(m),)).validate()

def test_checked_host_binding_reuses_only_complete_immutable_package(independent_primal_package,monkeypatch):
    import tessera.compiler.native_scaled_program as native
    monkeypatch.setenv("TESSERA_ROCM_PROGRAM_BINDING_CACHE","1")
    native._cached_scaled_abi.cache_clear()
    first=native._scaled_abi_binding(independent_primal_package)
    second=native._scaled_abi_binding(independent_primal_package)
    assert second is first
    assert native._cached_scaled_abi.cache_info().hits==1
    manifest=independent_primal_package.to_manifest()
    checked=native.NativeScaledProgram.from_manifest(manifest)
    assert native.NativeScaledProgram.from_manifest(manifest) is checked
    # Reusing the same mutable dictionary must not reuse checked wrong geometry.
    member=json.loads(manifest["members_json"][0]);member["geometry"][2]=2
    manifest["members_json"][0]=json.dumps(member)
    with pytest.raises(ValueError,match="batch"):
        native.NativeScaledProgram.from_manifest(manifest)

def test_host_binding_bypass_and_byte_admission(independent_primal_package,monkeypatch):
    import tessera.compiler.native_scaled_program as native
    monkeypatch.setenv("TESSERA_ROCM_PROGRAM_BINDING_CACHE","0")
    assert native._scaled_abi_binding(independent_primal_package) is not native._scaled_abi_binding(independent_primal_package)
    monkeypatch.setenv("TESSERA_ROCM_PROGRAM_BINDING_CACHE","1")
    monkeypatch.setattr(native,"_BINDING_ENTRY_BYTES",1)
    assert native._scaled_abi_binding(independent_primal_package) is not native._scaled_abi_binding(independent_primal_package)


def test_owner_diagnostic_mutation_cannot_rewrite_checked_storage(independent_primal_package):
    import numpy as np
    import tessera.compiler.native_scaled_program as native
    owner=native.PreparedScaledProgram.__new__(native.PreparedScaledProgram)
    owner.package=independent_primal_package
    owner._binding=native._scaled_abi_binding(independent_primal_package)
    owner._program=None
    # A compact-input owner still has its runtime provider before admission.
    from types import SimpleNamespace
    owner.lib=SimpleNamespace()
    values=[]
    for spec in owner._binding.storage[:owner._binding.argument_count]:
        dtype=np.float32 if spec.storage=="f32" else np.uint8
        values.append(np.zeros(spec.shape,dtype=dtype))
    owner._inputs(values)
    owner.program["buffers"][0]["shape"]=[1]
    owner.program["buffers"][0]["bytes"]=1
    owner.program["argument_count"]=1
    owner.program["outputs"]=[]
    # The real input frame remains valid; a forged one-byte shape is refused.
    owner._inputs(values)
    bad=list(values);bad[0]=np.zeros((1,),dtype=np.uint8)
    with pytest.raises(ValueError,match="storage/shape/layout"):
        owner._inputs(bad)
    assert owner._binding.outputs
    from types import SimpleNamespace
    import ctypes as c
    reads=[]
    owner.handle=c.c_uint64(1)
    owner.lib=SimpleNamespace(tessera_rocm_program_read=lambda handle,slot,generation,pointer,size: reads.append((slot,size)) or 0)
    outputs=owner.read(1)
    assert len(outputs)==len(owner._binding.outputs)
    assert [slot for slot,_ in reads]==list(owner._binding.outputs)
    assert [out.shape for out in outputs]==[
        owner._binding.storage[slot].shape for slot in owner._binding.outputs]
