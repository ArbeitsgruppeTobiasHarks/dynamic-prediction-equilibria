import os

from Cython.Build import cythonize
from setuptools import setup

setup(
    name='Predictor',
    ext_modules=cythonize(
        os.path.dirname(os.path.realpath(__file__)) + "/**/*.pyx",
        language_level='3',
        language='c++'),
    zip_safe=False,
)
