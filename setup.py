from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    ext_modules = cythonize("mesh_labeller/rasterize_tris.pyx"),
    include_dirs=[numpy.get_include()]
)