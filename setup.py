"""Build Script"""

import glob
import os
import platform

import numpy as np
import torch
from setuptools import setup, Extension
from torch.utils.cpp_extension import CUDA_HOME, CppExtension, CUDAExtension, ROCM_HOME

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT = "point_frustums"


def get_extensions() -> list[Extension]:
    """
    Get a list of extensions that shall be provided to the setup/build step.
    :return:
    """
    extensions_dir = os.path.join(PROJECT, "ops", "csrc")
    extensions_main = os.path.join(extensions_dir, "ext.cpp")

    # Gather .cpp sources (relative paths)
    extensions_sources = [
        os.path.relpath(p, BASE_DIR)
        for p in glob.glob(
            os.path.join(BASE_DIR, extensions_dir, "**", "*.cpp"), recursive=True
        )
    ]

    # Ensure no duplicates
    sources = [extensions_main] + [
        s for s in extensions_sources if not s.endswith("ext.cpp")
    ]

    is_rocm_pytorch = bool((torch.version.hip is not None) and (ROCM_HOME is not None))

    extensions_sources_cuda = [
        os.path.relpath(p, BASE_DIR)
        for p in glob.glob(
            os.path.join(BASE_DIR, extensions_dir, "**", "*.cu"), recursive=True
        )
        + glob.glob(os.path.join(BASE_DIR, extensions_dir, "*.cu"))
    ]
    extension = CppExtension

    extra_compile_args = {"cxx": []}
    define_macros = []

    if platform.system() == "Darwin":  # macOS
        extra_compile_args["cxx"].extend(
            [
                "-DTORCH_DISABLE_PARALLEL_NATIVE",
            ]
        )

    if (
        torch.cuda.is_available() and ((CUDA_HOME is not None) or is_rocm_pytorch)
    ) or os.getenv("FORCE_CUDA", "0") == "1":
        extension = CUDAExtension
        sources += extensions_sources_cuda

        if not is_rocm_pytorch:
            define_macros += [("WITH_CUDA", None)]
            extra_compile_args["nvcc"] = [
                "-O3",
                "-DCUDA_HAS_FP16=1",
                "-D__CUDA_NO_HALF_OPERATORS__",
                "-D__CUDA_NO_HALF_CONVERSIONS__",
                "-D__CUDA_NO_HALF2_OPERATORS__",
            ]
        else:
            define_macros += [("WITH_HIP", None)]
            extra_compile_args["nvcc"] = []

        nvcc_flags_env = os.getenv("NVCC_FLAGS", "")
        if nvcc_flags_env:
            extra_compile_args["nvcc"].extend(nvcc_flags_env.split())

    include_dirs = [os.path.join(BASE_DIR, extensions_dir)]

    ext_modules = [
        extension(
            "point_frustums._C",
            sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        )
    ]

    return ext_modules


# Build extensions only - all metadata is now in pyproject.toml.old
setup(
    ext_modules=get_extensions(),
    include_dirs=[np.get_include()],
    cmdclass={"build_ext": torch.utils.cpp_extension.BuildExtension},
)
