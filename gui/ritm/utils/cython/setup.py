#use 'python setup.py build_ext --inplace' to execute this script to compile _get_dist_maps.pyx for windows users.
from setuptools import setup
from Cython.Build import cythonize
import numpy as np

setup(
    ext_modules=cythonize("_get_dist_maps.pyx"),
    include_dirs=[np.get_include()]
)