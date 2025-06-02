from setuptools import setup, Extension
import numpy

# Define the CCG C extension
ccg_extension = Extension(
    'bombcell.ccg_heart',
    sources=['bombcell/ccg_heart.c'],
    include_dirs=[numpy.get_include()],
    extra_compile_args=['-O3', '-ffast-math'],
    language='c'
)

setup(ext_modules=[ccg_extension])