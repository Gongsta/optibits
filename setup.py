from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="custom_fp8",
    ext_modules=[
        CUDAExtension(
            "custom_fp8",
            ["custom_fp8.cu"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
