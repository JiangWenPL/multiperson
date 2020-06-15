from setuptools import setup, find_packages

from torch.utils.cpp_extension import BuildExtension, CUDAExtension

CUDA_FLAGS = []
INSTALL_REQUIREMENTS = []

ext_modules=[
    CUDAExtension('sdf.csrc', [
        'sdf/csrc/sdf_cuda.cpp',
        'sdf/csrc/sdf_cuda_kernel.cu',
        ]),
    ]

setup(
    description='PyTorch implementation of SDF loss',
    author='Nikos Kolotouros',
    author_email='nkolot@seas.upenn.edu',
    license='MIT License',
    version='0.0.1',
    name='sdf_pytorch',
    packages=['sdf', 'sdf.csrc'],
    install_requires=INSTALL_REQUIREMENTS,
    ext_modules=ext_modules,
    cmdclass = {'build_ext': BuildExtension}
)
