from pathlib import Path
import hashlib
import json
import os
import struct
from tessera.compiler.rocm_nvfp4_expansion_program import package_native_nvfp4_expansion_program
root=Path("benchmarks/baselines/nvfp4_bf16_store_source_20261017/original_checkpoint/original-store-source")
output=Path(os.environ["CAST_IMAGE_CONTROL_OUTPUT"])
def semantic_sections(data):
    shoff=struct.unpack_from("<Q",data,40)[0]
    size,count,names_index=struct.unpack_from("<HHH",data,58)
    headers=[struct.unpack_from("<IIQQQQIIQQ",data,shoff+i*size) for i in range(count)]
    header=headers[names_index]
    names=data[header[4]:header[4]+header[5]]
    result={}
    for row in headers:
        name=names[row[0]:].split(b"\0",1)[0].decode()
        content=data[row[4]:row[4]+row[5]]
        if name in (".text",".rodata"):
            result[name]=hashlib.sha256(content).hexdigest()
        if row[1]==7:
            pos=0
            while pos+12<=len(content):
                a,b,kind=struct.unpack_from("<III",content,pos);pos+=12
                owner=content[pos:pos+a].rstrip(b"\0");pos=(pos+a+3)&~3
                desc=content[pos:pos+b];pos=(pos+b+3)&~3
                if owner==b"AMDGPU":
                    result[f"AMDGPU_note_{kind}"]=hashlib.sha256(desc).hexdigest()
    return result
rows=[]
for m in (19,256):
    package=package_native_nvfp4_expansion_program((root/f"source-{m}.mlir").read_text())
    for i,p in enumerate(package.native_stages):
        output.with_name(f"{output.stem}-{m}-{i}.hsaco").write_bytes(p.image.payload)
    rows.append({"m":m,"recorder_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"members":[{"payload_sha256":hashlib.sha256(p.image.payload).hexdigest(),
        "target_ir_digest":p.image.target_ir_digest,"entry":p.descriptor.entry_symbol,
        "artifact_image_digest":p.image.image_digest,
        "semantic_sections":semantic_sections(p.image.payload),
        "device_libraries":[library.to_dict() for library in p.image.device_libraries]} for p in package.native_stages]})
output.write_text(json.dumps(rows,indent=2)+"\n")
print("recorded six existing member payload controls",flush=True)
