# Native GPU package and nested lifetime proof

Owner W2.4a / CAKE / SO-2; synchronization key `IR-NATIVE-FOUNDATION-1`.

The [package API](../python/tessera/compiler/native_gpu_storage.py) binds the
compiler-emitted sizing companion and GPU image into one immutable identity.
Reloading requires the expected digest. Kernel/sizer/ABI/image mutations fail
before loading; the loaded companion supplies bytes to the loaded kernel using
the same arguments. Invalid sizes return a native failure and raise before
launch. This is a synchronous raw device-pointer/index interface on the caller's
current context, with caller-owned allocations and kernel geometry preconditions.
It does not yet supply typed tensor descriptors or automatic JIT/arbiter dispatch.

Both owning WSL hosts pass serialized-package execution for nested scratch at
32/64/128/256 threads and 1/7/17/33 iterations, respectively, with 32 blocks.
Each iteration publishes thread-local values into scratch, reads the neighboring
lane and releases the storage before the backedge. Every output matches exactly.

RTX 5070 additionally passes those four cases with a real registered NVGPU
asynchronous copy from global memory. The direct copy token enters a commit
group, a full group wait completes it, and a GPU barrier publishes the result.
Missing waits, partial waits and unrelated group tokens fail before codegen.
These tests prove producer integration and completion; they do not benchmark
compute overlap or integrate the separate macro-GEMM schedule's release tokens.
ROCm has nested-package proof but still needs its own asynchronous producer.
Apple requires a separate MSL/Metal binding. No sibling performance claim follows.

- [NVIDIA evidence](baselines/native_gpu_package_nvidia.json)
- [ROCm evidence](baselines/native_gpu_package_rocm.json)
- [Recorder](record_native_gpu_storage_package.py)

The packets record device identity, compiler/fixture/recorder/image/host-library
hashes and package binding digests. Oversized launches are rejected before GPU
dispatch. No timing or sanitizer claim is made. CUDA native disassembly is
confirmed `LDGSTS.E`, `LDGDEPBAR` and `DEPBAR.LE` in the
[measured async image](baselines/native_gpu_package_nvidia_async.sass).

Reproduce on the owning host with its device visible and native compiler built:

```sh
PYTHONPATH=python .venv/bin/python benchmarks/record_native_gpu_storage_package.py \
  --backend rocm --compiler build/tools/tessera-opt/tessera-opt \
  --artifacts /tmp/tessera-native-package-rocm \
  --output /tmp/native_gpu_package_rocm.json
```

For CUDA use `--backend nvidia`, the selected core compiler, CUDA 13.3 and
`scripts/_nvidia_env.sh`. The artifact directory retains serialized packages and
native images. Never load an untrusted native package merely because its
self-reported hash matches; the caller must pin the expected package identity.
