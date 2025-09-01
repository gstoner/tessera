import os
from pathlib import Path
from torch.utils.cpp_extension import CUDAExtension, BuildExtension, setup

target_sm = os.environ.get("TESSERA_TARGET_SM","90")
extra_cuda_cflags = [
    f"-gencode=arch=compute_{target_sm},code=sm_{target_sm}",
    "-lineinfo",
    "-O3",
    "-UDEBUG"
]

if os.environ.get("TESSERA_ENABLE_BF16","0") == "1":
    extra_cuda_cflags.append("-DTESSERA_ENABLE_BF16")

sources = [
    "tessera_ext/flashattn_bwd_tiled.cu",
    "tessera_ext/flashattn_fwd_tiled.cu",
    "tessera_ext/binding.cpp",
    "tessera_ext/gemm_wmma.cu",
    "tessera_ext/gemm_wmma_bf16.cu",
    "tessera_ext/reduce_tile.cu",
    "tessera_ext/flashattn_naive_fwd.cu",
    "tessera_ext/flashattn_bwd_fused.cu",
]
include_dirs = [str(Path(__file__).parent / "tessera_ext")]

setup(
    name="tessera_ext",
    ext_modules=[
        CUDAExtension(
            name="tessera_ext",
            sources=sources,
            include_dirs=include_dirs,
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": extra_cuda_cflags
            }
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    zip_safe=False,
)
