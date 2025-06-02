"""
Setup script for building the CCG C extension
Run with: python ccg_setup.py build_ext --inplace
"""

from setuptools import setup, Extension
import numpy

# Define the extension module
ccg_extension = Extension(
    'ccg_heart',
    sources=['ccg_heart.c'],
    include_dirs=[numpy.get_include()],
    extra_compile_args=['-O3', '-ffast-math'],  # Optimization flags
    language='c'
)

setup(
    name='ccg_heart',
    ext_modules=[ccg_extension],
    zip_safe=False,
)