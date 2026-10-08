"""Record LLVM's maximum register-pressure report for the native folded route.

This is compiler attribution, not a hardware counter or a Python kernel author.
The probe translates only the LLVM/ROCDL dialect emitted before native GPU
serialization and compares its selected instructions/resources to that image.
"""
from __future__ import annotations
import argparse
from collections import Counter
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/"python")]
from tessera import runtime as rt
from tessera.compiler.rocm_mxfp4_folded_frontend import author_folded_scaled_matmul_graph
from tessera.compiler.rocm_native import _shape_free_target_ir,_extract_hsaco
from tessera.compiler.rocm_pipeline import ROCMExecutablePipeline,ROCMInputLevel,ROCMOutputLevel
from benchmarks.rocm.benchmark_gfx1201_mxfp4_production import _code_object_evidence
from benchmarks.rocm.inspect_gfx1201_folded_prefill import selected_symbol_isa_evidence

def run(command,*,source=None):
    result=subprocess.run([str(x) for x in command],input=source,
                          text=True,capture_output=True)
    if result.returncode:
        raise RuntimeError(f"{command[0]} failed: {result.stderr[-4000:]}")
    return result

def pressure_summary(report):
    vg=report.split("*** Register pressure info (SGPRs)",1)[0]
    maximum=re.search(r"Max pressure is (\d+) VGPRs at (.+)",vg)
    if not maximum:
        raise RuntimeError("LLVM did not report maximum VGPR pressure")
    counts=Counter()
    registers=[]
    for block in re.split(r"(?m)^  (?=%\d+:)",vg)[1:]:
        header=block.splitlines()[0]
        width=re.search(r"\((\d+) VGPRs\)",header)
        if not width: continue
        definitions=[x.strip() for x in block.splitlines()
                     if x.strip().startswith("def ")]
        if any("WMMA" in x for x in definitions): category="wmma_accumulator"
        elif any("LOAD_DWORDX4" in x for x in definitions): category="global_prefetch_vectors"
        elif any("DS_READ" in x for x in definitions): category="lds_fragment_vectors"
        else: category="other_address_or_state"
        count=int(width[1])
        counts[category]+=count
        registers.append({"header":header,"vgprs":count,"category":category,
                          "definitions":definitions})
    # An early-clobber instruction can need its result in addition to the
    # report's listed live-in set. Preserve this difference, never assign
    # unattributed pressure to a guessed source category.
    listed=sum(counts.values())
    if listed>int(maximum[1]):
        raise RuntimeError("LLVM register report accounting changed")
    return {"max_vgprs":int(maximum[1]),"location":maximum[2],
            "listed_live_vgprs":listed,
            "unattributed_at_instruction_vgprs":int(maximum[1])-listed,
            "categories":dict(counts),"registers":registers}

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--tessera-opt",required=True,type=Path)
    parser.add_argument("--llvm-bin",required=True,type=Path)
    parser.add_argument("--output-dir",required=True,type=Path)
    args=parser.parse_args()
    if rt._rocm_live_arch()!="gfx1201":
        raise RuntimeError("folded pressure attribution requires actual gfx1201")
    os.environ["TESSERA_OPT"]=str(args.tessera_opt.resolve())
    out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
    graph=author_folded_scaled_matmul_graph(256,4096,1024)
    target=run([args.tessera_opt,"--tessera-graph-to-schedule",
                "--tessera-schedule-to-tile","--lower-tile-to-rocm=arch=gfx1201"],
               source=graph).stdout
    target=_shape_free_target_ir(target,family="folded_matmul",
        directive="tessera_rocm.scaled_wmma_gemm",runtime_k=True)
    config=ROCMExecutablePipeline(family="matmul",arch="gfx1201",
                                 input_level=ROCMInputLevel.DIRECTIVE)
    native=run([args.tessera_opt,"--pass-pipeline="+
        config.pass_pipeline(output=ROCMOutputLevel.BINARY),
        "--mlir-print-ir-before=gpu-module-to-binary","--mlir-disable-threading"],
        source=target)
    lines=native.stderr.rstrip().splitlines()
    gpu=[i for i,line in enumerate(lines) if "gpu.module @" in line]
    if len(gpu)!=1 or lines[-1]!="}" or lines[-2].strip()!="}":
        raise RuntimeError("native pre-serialization dump structure changed")
    index=gpu[0]
    layout=re.search(r'llvm.data_layout = "([^"]+)"',lines[index])
    if layout is None or not lines[index-1].startswith("module "):
        raise RuntimeError("native GPU data-layout/module boundary missing")
    # Flatten the sole gpu.module for the upstream LLVM dialect translator.
    # This is diagnostic extraction only, never an executable package route.
    dialect="\n".join(lines[1:index-1]+[
        'module attributes {llvm.data_layout = "'+layout[1]+'"} {'
        ]+lines[index+1:-1])+"\n"
    (out/"llvm-dialect.mlir").write_text(dialect)
    llvm=run([args.llvm_bin/"mlir-translate","--mlir-to-llvmir",
              out/"llvm-dialect.mlir"]).stdout
    (out/"kernel.ll").write_text(llvm)
    flags=["-mtriple=amdgcn-amd-amdhsa","-mcpu=gfx1201",
           "-mattr=+wavefrontsize32,-wavefrontsize64","-O2"]
    run([args.llvm_bin/"opt",*flags,"-S",out/"kernel.ll",
         "-o",out/"kernel-opt.ll"])
    optimized_llvm=(out/"kernel-opt.ll").read_text()
    branch_weights=[line.strip() for line in optimized_llvm.splitlines()
                    if '"branch_weights"' in line]
    pressure=run([args.llvm_bin/"llc",*flags,
        "-amdgpu-print-max-reg-pressure-regusage-after-scheduler",
        "-filetype=obj",out/"kernel-opt.ll","-o",out/"probe.o"]).stderr
    payload=_extract_hsaco(native.stdout)
    entry=re.search(r'name = "([^"]+)"',target)[1]
    actual=selected_symbol_isa_evidence(payload,entry)
    probe=(out/"probe.o").read_bytes()
    diagnostic=selected_symbol_isa_evidence(probe,entry)
    actual_resources=_code_object_evidence(payload)["resources"]
    probe_resources=_code_object_evidence(probe)["resources"]
    receipt={"schema":"tessera.rocm.folded_register_pressure.v1",
        "architecture":rt._rocm_live_arch(),"shape_mnk":[256,4096,1024],
        "physical_contract":"rocm_mxfp4_w4a8_folded_prefill_v1",
        "native_isa":actual,"diagnostic_isa":diagnostic,
        "native_resources":actual_resources,"diagnostic_resources":probe_resources,
        "selected_instruction_stream_matches_native":
            actual["instruction_stream_sha256"]==diagnostic["instruction_stream_sha256"],
        "pressure":pressure_summary(pressure),"llc_flags":flags,
        "optimized_branch_weight_nodes":branch_weights,
        "llvm_version":run([args.llvm_bin/"llc","--version"]).stdout.splitlines()[0],
        "compiler_sha256":hashlib.sha256(args.tessera_opt.read_bytes()).hexdigest(),
        "target_sha256":hashlib.sha256(target.encode()).hexdigest(),
        "source_sha256":{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
            for name in ["benchmarks/rocm/record_gfx1201_folded_register_pressure.py",
              "src/compiler/codegen/Tessera_ROCM_Backend/lib/Conversion/GenerateWMMAGemmKernel.cpp",
              "src/compiler/codegen/Tessera_ROCM_Backend/lib/Conversion/TileToROCM.cpp"]},
        "scope":"compiler after-scheduler virtual liveness; not physical allocation or hardware counters"}
    (out/"pressure.json").write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n")
    (out/"target.mlir").write_text(target)
    for filename in ["llvm-dialect.mlir","kernel.ll","kernel-opt.ll"]:
        path=out/filename
        with gzip.open(str(path)+".gz","wb") as stream:stream.write(path.read_bytes())
        path.unlink()
    with gzip.open(out/"pressure.log.gz","wb") as stream:stream.write(pressure.encode())
    (out/"probe.o").unlink()
    print(json.dumps({key:receipt[key] for key in [
        "native_resources","diagnostic_resources","selected_instruction_stream_matches_native"]}))
    print(receipt["pressure"]["max_vgprs"],receipt["pressure"]["categories"])

if __name__=="__main__":main()
