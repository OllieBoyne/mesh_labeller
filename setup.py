from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    ext_modules = cythonize("rasterize_tris.pyx"),
    include_dirs=[numpy.get_include()]
)