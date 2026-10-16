"""Reproduce bounded NVFP4 source owner event/member/completed-host scopes."""
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import statistics
import time

import numpy as np
from tessera import runtime as rt
from tessera.compiler.rocm_nvfp4_expansion_program import package_native_nvfp4_expansion_program
from tests.device.rocm.test_prepared_nvfp4_expansion_program import inputs_and_oracle
from tests.unit.test_native_nvfp4_expansion_program import source_graph

w = Path("/home/angstorms/scratch/nvfp4-source-bounded-20261016")
assert rt._rocm_live_arch() == "gfx1201"
hip = rt._load_hip_for_launch()
device, name = C.c_int(), C.create_string_buffer(256)
assert hip.hipGetDevice(C.byref(device)) == 0
assert hip.hipDeviceGetName(name, len(name), device.value) == 0 and name.value
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
record = {"schema": "tessera.nvfp4_source_bounded_characterization.v1",
          "architecture": rt._rocm_live_arch(), "device": name.value.decode(),
          "compiler_sha256": sha(os.environ["TESSERA_OPT"]),
          "rocm_tool_sha256": sha(os.environ["TESSERA_ROCM_OPT"]),
          "provider_sha256": sha(os.environ["TESSERA_ROCM_MULTI_OUTPUT_LIB"]),
          "recorder_sha256": sha(__file__), "model_quality_acceptance": False,
          "timing_scope": "grouped member graphs include device graph dispatch; program HIP event windows may include host enqueue idle; completed prepared invoke/read excludes prepare and upload. No pure-kernel or public-JIT timing claim.",
          "rows": []}
for cap,n,k,output in ((64,7,48,"f32"), (64,7,48,"bf16"), (256,4096,16,"f32")):
    source = source_graph(19,n,k,output=output).replace(
        'tessera.arch = "gfx1201"', f'tessera.arch = "gfx1201", tessera.native.nvfp4_m_bound = {cap} : i64')
    package = package_native_nvfp4_expansion_program(source)
    roots, expected = inputs_and_oracle(19,n,k,tuple(range(5)),output)
    with package.prepare(roots, runtime_library=os.environ["TESSERA_ROCM_MULTI_OUTPUT_LIB"]) as owner:
        initial=owner.rows_info()
        owner.import_weights()
        for m in (19,1,cap):
            index=np.arange(m)%19
            owner.update_activations(roots[3][index].copy(),roots[4][index].copy())
            wanted=expected[index]
            generation,_=owner.invoke_consumer()
            np.testing.assert_array_equal(owner.read(generation)[0],wanted)
            members,events,hosts,member_windows,event_windows,host_windows=[],[],[],[],[],[]
            repeats,host_repeats=8192,256
            for _ in range(7):
                generation,stage_times=owner.profile_members(repeats=repeats)
                np.testing.assert_array_equal(owner.read(generation)[0],wanted)
                durations=[t*repeats for t in stage_times]
                assert min(durations)>=10., durations
                members.append(stage_times);member_windows.append(durations)
                generation,event=owner.invoke(repeats=repeats,timed=True)
                np.testing.assert_array_equal(owner.read(generation)[0],wanted)
                assert event*repeats>=10.
                events.append(event);event_windows.append(event*repeats)
                start=time.perf_counter_ns()
                last=None
                for _ in range(host_repeats):
                    generation,_=owner.invoke()
                    last=owner.read(generation)[0]
                duration=(time.perf_counter_ns()-start)/1e6
                assert duration>=10.
                np.testing.assert_array_equal(last,wanted)
                hosts.append(duration/host_repeats);host_windows.append(duration)
                assert owner.rows_info()==(cap,m,*initial[2:])
            row={"capacity_m":cap,"shape_mnk":[m,n,k],"output_storage":output,
                 "staging":package.consumer.descriptor.provenance["staging"],
                 "binding_digest":package.binding_digest,
                 "images":[package.producer.native.image.image_digest,package.consumer.image.image_digest],
                 "allocation_bytes":initial[2],"allocation_count":initial[3],
                 "device_repeats":repeats,"host_repeats":host_repeats,
                 "member_grouped_dispatch_mean_samples_ms":members,
                 "program_event_mean_samples_ms":events,"prepared_invoke_read_mean_samples_ms":hosts,
                 "member_window_duration_ms":member_windows,"program_window_duration_ms":event_windows,
                 "host_window_duration_ms":host_windows,
                 "correctness":"independent nibble decode FP64 oracle; bitwise before and after every timing window",
                 "member_median_ms":[statistics.median(x[i] for x in members) for i in (0,1)],
                 "program_median_ms":statistics.median(events),"prepared_host_median_ms":statistics.median(hosts)}
            record["rows"].append(row)
            (w/"bounded-source-timing.json").write_text(json.dumps(record,indent=2)+"\n")
            print(json.dumps({key:row[key] for key in ("shape_mnk","output_storage","member_median_ms","program_median_ms","prepared_host_median_ms")}),flush=True)
