from setuptools import setup, Extension
from Cython.Build import cythonize

ext = Extension(
    name="game_ai",
    sources=["game_ai.pyx"],
    language="c++",
    extra_compile_args=["/std:c++17"],
)


setup(
    name="game_ai",
    ext_modules=cythonize(ext),
)
# python setup.py build_ext --inplace