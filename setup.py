from setuptools import setup, Extension
import numpy

setup(
    ext_modules = Extension("mesh_labeller/rasterize_tris.pyx"),
    include_dirs=[numpy.get_include()],
    setup_requires=["cython"],
    install_requires=["imageio", "numpy", "pyrender", "PyYAML", "trimesh", "pyembree", "opencv-python"]
)