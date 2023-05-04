from setuptools import setup, Extension

class get_numpy_include(object):
    """Workaround to postpone numpy import until after setuptools finishes installing dependencies."""
    def __str__(self):
        import numpy
        return numpy.get_include()

setup(
    ext_modules = [Extension(name='rasterize_tris', sources=["mesh_labeller/rasterize_tris.pyx"])],
    include_dirs=[get_numpy_include()],
    setup_requires=["numpy", "cython"],
    install_requires=["imageio", "numpy", "pyrender", "PyYAML", "trimesh", "pyembree", "opencv-python"]
)