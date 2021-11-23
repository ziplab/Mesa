# Copyright (c) 2021-present, Zhuang AI Group.
# All rights reserved.

import torch
from setuptools import setup, Extension, find_packages
from torch.utils import cpp_extension

compile_args = {"cxx": [], "nvcc": [] }

if torch.__version__ < "1.8":
    version = torch.__version__.split('.')
    compile_args['cxx'] += ["-DTORCH_VERSION_MAJOR={}".format(version[0])]
    compile_args['cxx'] += ["-DTORCH_VERSION_MINOR={}".format(version[1])]

setup(
        name = 'Mesa',
        version = '1.0',
        packages=find_packages(),
        ext_modules=[
            cpp_extension.CppExtension(
            'mesa.native',
            ['native.cpp'],
            extra_compile_args=compile_args,
            ),
            cpp_extension.CUDAExtension(
                'mesa.cpp_extension.quantization',
                ['mesa/cpp_extension/quantization.cc',
                 'mesa/cpp_extension/quantization_cuda_kernel.cu']
            ),
        ],
        cmdclass={'build_ext': cpp_extension.BuildExtension})

