from setuptools import setup, Extension, find_packages
import os

# Try to import numpy, but handle gracefully if not available during build
try:
    import numpy
    numpy_include = numpy.get_include()
except ImportError:
    # Fallback for when numpy is not yet installed
    numpy_include = ''

# Define the CCG C extension
ccg_extension = Extension(
    'bombcell.ccg_heart',
    sources=['bombcell/ccg_heart.c'],
    include_dirs=[numpy_include] if numpy_include else [],
    extra_compile_args=['-O3', '-ffast-math'],
    language='c'
)

# Read requirements
def read_requirements():
    try:
        req_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'requirements.txt')
        if os.path.exists(req_file):
            with open(req_file, 'r') as f:
                return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    except:
        pass
    return [
        'numpy>=1.19.0',
        'scipy>=1.7.0',
        'matplotlib>=3.3.0',
        'pandas>=1.3.0',
        'ipywidgets>=7.6.0'
    ]

# Read version
def get_version():
    try:
        version_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bombcell', '__init__.py')
        if os.path.exists(version_file):
            with open(version_file, 'r') as f:
                for line in f:
                    if line.startswith('__version__'):
                        return line.split('=')[1].strip().strip('"\'')
    except:
        pass
    return '0.1.0'

# Read README
def read_readme():
    try:
        readme_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'README.md')
        if os.path.exists(readme_file):
            with open(readme_file, 'r', encoding='utf-8') as f:
                return f.read()
    except:
        pass
    return 'Python implementation of BombCell quality metrics for spike sorting'

setup(
    name='bombcell',
    version=get_version(),
    description='Python implementation of BombCell quality metrics for spike sorting',
    long_description=read_readme(),
    long_description_content_type='text/markdown',
    author='BombCell Team',
    packages=find_packages(),
    install_requires=read_requirements(),
    ext_modules=[ccg_extension] if numpy_include else [],
    python_requires='>=3.8',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
    include_package_data=True,
    zip_safe=False,
)