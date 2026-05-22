from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="tessera_kernels",
    ext_modules=[
        CUDAExtension(
            name="tessera_kernels",
            sources=["csrc/tessera_kernels.cu"],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "--use_fast_math"]
            }
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    zip_safe=False,
)
